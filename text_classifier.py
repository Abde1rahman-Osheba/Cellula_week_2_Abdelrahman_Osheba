"""
text_classifier.py
------------------
Text classification module implementing a hybrid ensemble of:
  1. Fine-tuned DistilBERT with LoRA
  2. Fine-tuned ALBERT with LoRA
  3. Bidirectional LSTM

A LogisticRegression meta-model combines the predictions from all three
base models for the final classification.

If pre-saved artifacts exist in ./toxic_hybrid_artifacts/, the module loads
them directly. Otherwise, it trains all models from scratch (requires GPU
and the dataset CSV).
"""

import os
import re
import json
import random
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

from peft import LoraConfig, get_peft_model, PeftModel, TaskType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DISTIL_BASE = "distilbert-base-uncased"
ALBERT_BASE = "albert-base-v2"

DISTIL_OUT = "./distilbert_lora_toxic"
ALBERT_OUT = "./albert_lora_toxic"

ARTIFACT_DIR = Path("./toxic_hybrid_artifacts")
DATA_PATH = Path("./cellula toxic data.csv")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)


# ===========================================================================
# Helper Dataset / Tokenization utilities
# ===========================================================================

@dataclass
class HFTextDataset(Dataset):
    """Simple HuggingFace-compatible dataset for text classification."""
    texts: List[str]
    labels: List[int]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "labels": self.labels[idx]}


def build_hf_dataset(df_part: pd.DataFrame) -> "HFTextDataset":
    return HFTextDataset(df_part["text"].tolist(), df_part["label_id"].tolist())


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


# ===========================================================================
# LSTM helpers
# ===========================================================================

def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\[\]sep]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split(" ")


def build_vocab(texts: List[str], min_freq: int = 2, max_vocab: int = 30000):
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.most_common():
        if freq < min_freq:
            continue
        if word in vocab:
            continue
        vocab[word] = len(vocab)
        if len(vocab) >= max_vocab:
            break
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int = 128) -> List[int]:
    tokens = simple_tokenize(text)
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens][:max_len]
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


class LSTMDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(
            encode_text(self.texts[idx], self.vocab, self.max_len), dtype=torch.long
        )
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class ToxicLSTM(nn.Module):
    """Bidirectional LSTM for text classification."""

    def __init__(
        self, vocab_size, num_labels, emb_dim=128, hidden=128, dropout=0.3, bidir=True
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=bidir,
        )
        out_dim = hidden * (2 if bidir else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_labels)

    def forward(self, x):
        emb = self.embedding(x)
        out, (h, c) = self.lstm(emb)
        if self.lstm.bidirectional:
            last = torch.cat((h[-2], h[-1]), dim=1)
        else:
            last = h[-1]
        last = self.dropout(last)
        logits = self.fc(last)
        return logits


# ===========================================================================
# Training functions
# ===========================================================================

def train_lora_transformer(
    base_model_name: str,
    output_dir: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    num_labels: int,
    lr: float = 2e-4,
    epochs: int = 4,
    batch_size: int = 16,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
):
    """Fine-tune a transformer with LoRA for sequence classification."""
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=num_labels
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=lora_target_modules if lora_target_modules else ["q_lin", "v_lin"],
    )
    model = get_peft_model(model, lora_config)

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    train_ds = build_hf_dataset(train_df)
    eval_ds = build_hf_dataset(eval_df)

    class TokenizingWrapper(Dataset):
        def __init__(self, base_ds):
            self.base_ds = base_ds

        def __len__(self):
            return len(self.base_ds)

        def __getitem__(self, idx):
            item = self.base_ds[idx]
            tok = tokenize_batch({"text": item["text"]})
            tok["labels"] = item["labels"]
            return tok

    train_tok = TokenizingWrapper(train_ds)
    eval_tok = TokenizingWrapper(eval_ds)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir


def train_lstm_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_labels: int,
    max_len: int = 128,
    batch_size: int = 64,
    lr: float = 1e-3,
    epochs: int = 8,
):
    """Train a bidirectional LSTM classifier."""
    vocab = build_vocab(train_df["text"].tolist(), min_freq=2, max_vocab=30000)

    train_ds = LSTMDataset(
        train_df["text"].tolist(), train_df["label_id"].tolist(), vocab, max_len=max_len
    )
    val_ds = LSTMDataset(
        val_df["text"].tolist(), val_df["label_id"].tolist(), vocab, max_len=max_len
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ToxicLSTM(vocab_size=len(vocab), num_labels=num_labels).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = -1
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_true.extend(yb.cpu().numpy().tolist())

        f1m = f1_score(all_true, all_preds, average="macro")
        print(
            f"Epoch {ep}/{epochs} | train_loss={np.mean(tr_losses):.4f} | val_f1_macro={f1m:.4f}"
        )

        if f1m > best_f1:
            best_f1 = f1m
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, vocab


# ===========================================================================
# Inference helpers
# ===========================================================================

@torch.no_grad()
def transformer_proba(
    model, tokenizer, texts: List[str], batch_size: int = 32
) -> np.ndarray:
    """Get class probabilities from a transformer model."""
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, truncation=True, max_length=256, padding=True, return_tensors="pt"
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


@torch.no_grad()
def lstm_proba(
    model: nn.Module,
    vocab: Dict[str, int],
    texts: List[str],
    max_len: int = 128,
    batch_size: int = 128,
) -> np.ndarray:
    """Get class probabilities from the LSTM model."""
    model.eval()
    probs_list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        x = torch.tensor(
            [encode_text(t, vocab, max_len=max_len) for t in batch], dtype=torch.long
        ).to(DEVICE)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        probs_list.append(probs)
    return np.vstack(probs_list)


def build_meta_features(
    texts: List[str],
    p_distil: np.ndarray,
    p_albert: np.ndarray,
    p_lstm: np.ndarray,
) -> np.ndarray:
    """Concatenate base-model probabilities into meta features."""
    return np.hstack([p_distil, p_albert, p_lstm])


# ===========================================================================
# Model loading
# ===========================================================================

def load_peft_seqcls(model_dir: str, base_model_name: str, num_labels: int):
    """Load a fine-tuned PEFT (LoRA) model and its tokenizer."""
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    base = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=num_labels
    )
    model = PeftModel.from_pretrained(base, model_dir)
    model.to(DEVICE)
    model.eval()
    return model, tok


# ===========================================================================
# TextClassifier — main public interface
# ===========================================================================

class TextClassifier:
    """
    Hybrid ensemble text classifier.

    On initialization, either loads pre-saved artifacts or trains all models
    from scratch (requires the dataset CSV and a GPU).
    """

    def __init__(self):
        self.label_names: List[str] = []
        self.num_labels: int = 0
        self.distil_model = None
        self.distil_tok = None
        self.albert_model = None
        self.albert_tok = None
        self.lstm_model = None
        self.lstm_vocab: Dict[str, int] = {}
        self.meta = None

        if self._artifacts_exist():
            self._load_artifacts()
        else:
            self._train_all()

    # -----------------------------------------------------------------------
    # Artifact check / load / save
    # -----------------------------------------------------------------------

    def _artifacts_exist(self) -> bool:
        """Check whether all pre-saved artifacts are available."""
        required = [
            ARTIFACT_DIR / "label_classes.json",
            ARTIFACT_DIR / "lstm.pt",
            ARTIFACT_DIR / "lstm_vocab.json",
            ARTIFACT_DIR / "meta_model.joblib",
            Path(DISTIL_OUT) / "adapter_config.json",
            Path(ALBERT_OUT) / "adapter_config.json",
        ]
        return all(p.exists() for p in required)

    def _load_artifacts(self):
        """Load pre-trained models and meta-model from disk."""
        print("[TextClassifier] Loading pre-saved artifacts …")

        self.label_names = json.loads(
            (ARTIFACT_DIR / "label_classes.json").read_text(encoding="utf-8")
        )
        self.num_labels = len(self.label_names)

        # DistilBERT + LoRA
        self.distil_model, self.distil_tok = load_peft_seqcls(
            DISTIL_OUT, DISTIL_BASE, self.num_labels
        )

        # ALBERT + LoRA
        self.albert_model, self.albert_tok = load_peft_seqcls(
            ALBERT_OUT, ALBERT_BASE, self.num_labels
        )

        # LSTM
        self.lstm_vocab = json.loads(
            (ARTIFACT_DIR / "lstm_vocab.json").read_text(encoding="utf-8")
        )
        self.lstm_model = ToxicLSTM(
            vocab_size=len(self.lstm_vocab), num_labels=self.num_labels
        ).to(DEVICE)
        self.lstm_model.load_state_dict(
            torch.load(ARTIFACT_DIR / "lstm.pt", map_location=DEVICE)
        )
        self.lstm_model.eval()

        # Meta-model
        self.meta = joblib.load(ARTIFACT_DIR / "meta_model.joblib")

        print("[TextClassifier] All models loaded successfully.")

    def _save_artifacts(self):
        """Save trained model artifacts to disk."""
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

        (ARTIFACT_DIR / "label_classes.json").write_text(
            json.dumps(self.label_names, indent=2), encoding="utf-8"
        )
        torch.save(self.lstm_model.state_dict(), ARTIFACT_DIR / "lstm.pt")
        (ARTIFACT_DIR / "lstm_vocab.json").write_text(
            json.dumps(self.lstm_vocab, indent=2), encoding="utf-8"
        )
        joblib.dump(self.meta, ARTIFACT_DIR / "meta_model.joblib")

        print(f"[TextClassifier] Artifacts saved to {ARTIFACT_DIR}")

    # -----------------------------------------------------------------------
    # Full training pipeline
    # -----------------------------------------------------------------------

    def _train_all(self):
        """Train all models from scratch using the dataset CSV."""
        print("[TextClassifier] Training all models from scratch …")

        # Load dataset
        df = pd.read_csv(DATA_PATH)
        df = df.rename(
            columns={
                "query": "query",
                "image descriptions": "image_desc",
                "Toxic Category": "label",
            }
        )
        df["query"] = df["query"].fillna("").astype(str)
        df["image_desc"] = df["image_desc"].fillna("").astype(str)
        df["label"] = df["label"].fillna("Unknown").astype(str)
        df["text"] = (
            df["query"].str.strip() + " [SEP] " + df["image_desc"].str.strip()
        ).str.strip()

        le = LabelEncoder()
        df["label_id"] = le.fit_transform(df["label"])
        self.label_names = list(le.classes_)
        self.num_labels = len(self.label_names)

        # Splits
        train_df, temp_df = train_test_split(
            df, test_size=0.30, random_state=SEED, stratify=df["label_id"]
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label_id"]
        )

        print(f"  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

        # DistilBERT + LoRA
        distil_dir = train_lora_transformer(
            base_model_name=DISTIL_BASE,
            output_dir=DISTIL_OUT,
            train_df=train_df,
            eval_df=val_df,
            num_labels=self.num_labels,
            lr=2e-4,
            epochs=4,
            batch_size=16,
        )

        # ALBERT + LoRA
        albert_dir = train_lora_transformer(
            base_model_name=ALBERT_BASE,
            output_dir=ALBERT_OUT,
            train_df=train_df,
            eval_df=val_df,
            num_labels=self.num_labels,
            lr=2e-4,
            epochs=4,
            batch_size=16,
            lora_target_modules=["query", "value"],
        )

        # LSTM
        self.lstm_model, self.lstm_vocab = train_lstm_model(
            train_df=train_df,
            val_df=val_df,
            num_labels=self.num_labels,
            max_len=128,
            batch_size=64,
            lr=1e-3,
            epochs=10,
        )

        # Load transformer models for inference
        self.distil_model, self.distil_tok = load_peft_seqcls(
            distil_dir, DISTIL_BASE, self.num_labels
        )
        self.albert_model, self.albert_tok = load_peft_seqcls(
            albert_dir, ALBERT_BASE, self.num_labels
        )

        # Meta-model (Logistic Regression on validation set)
        val_texts = val_df["text"].tolist()
        val_p_distil = transformer_proba(self.distil_model, self.distil_tok, val_texts)
        val_p_albert = transformer_proba(self.albert_model, self.albert_tok, val_texts)
        val_p_lstm = lstm_proba(self.lstm_model, self.lstm_vocab, val_texts)

        X_meta_val = build_meta_features(val_texts, val_p_distil, val_p_albert, val_p_lstm)
        y_meta_val = val_df["label_id"].to_numpy()

        self.meta = LogisticRegression(max_iter=2000, multi_class="multinomial")
        self.meta.fit(X_meta_val, y_meta_val)

        # Evaluate on test set
        test_texts = test_df["text"].tolist()
        test_p_distil = transformer_proba(self.distil_model, self.distil_tok, test_texts)
        test_p_albert = transformer_proba(self.albert_model, self.albert_tok, test_texts)
        test_p_lstm = lstm_proba(self.lstm_model, self.lstm_vocab, test_texts)

        X_meta_test = build_meta_features(
            test_texts, test_p_distil, test_p_albert, test_p_lstm
        )
        y_test = test_df["label_id"].to_numpy()
        test_preds = np.argmax(self.meta.predict_proba(X_meta_test), axis=1)

        print("\n=== HYBRID TEST RESULTS ===")
        print("Accuracy:", accuracy_score(y_test, test_preds))
        print("F1 macro:", f1_score(y_test, test_preds, average="macro"))
        print(
            "\nClassification report:\n",
            classification_report(y_test, test_preds, target_names=self.label_names),
        )

        # Save artifacts for future fast loading
        self._save_artifacts()

    # -----------------------------------------------------------------------
    # Public inference method
    # -----------------------------------------------------------------------

    def classify(self, query: str, image_caption: str = "") -> Dict:
        """
        Classify text using the hybrid ensemble.

        Parameters
        ----------
        query : str
            The user's text input.
        image_caption : str, optional
            An image caption to include in the classification.

        Returns
        -------
        dict
            {"label": str, "confidence": float, "probs": {label: float, ...}}
        """
        text = (str(query).strip() + " [SEP] " + str(image_caption).strip()).strip()

        p_d = transformer_proba(self.distil_model, self.distil_tok, [text])[0]
        p_a = transformer_proba(self.albert_model, self.albert_tok, [text])[0]
        p_l = lstm_proba(self.lstm_model, self.lstm_vocab, [text])[0]

        X = build_meta_features(
            [text], p_d.reshape(1, -1), p_a.reshape(1, -1), p_l.reshape(1, -1)
        )
        probs = self.meta.predict_proba(X)[0]
        pred_id = int(np.argmax(probs))
        conf = float(np.max(probs))

        return {
            "label": self.label_names[pred_id],
            "confidence": conf,
            "probs": {
                self.label_names[i]: float(probs[i]) for i in range(self.num_labels)
            },
        }
