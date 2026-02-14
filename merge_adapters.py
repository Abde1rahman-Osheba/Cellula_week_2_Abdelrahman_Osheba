"""
Merges LoRA adapters into base models and saves as standard HuggingFace models.
Usage: python merge_adapters.py
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DISTIL_BASE = "distilbert-base-uncased"
ALBERT_BASE = "albert-base-v2"

DISTIL_ADAPTER_DIR = "./distilbert_lora_toxic"
ALBERT_ADAPTER_DIR = "./albert_lora_toxic"

DISTIL_MERGED_DIR = "./distilbert_merged"
ALBERT_MERGED_DIR = "./albert_merged"

ARTIFACT_DIR = Path("./toxic_hybrid_artifacts")


def merge_and_save(adapter_dir: str, base_model_name: str, merged_dir: str, num_labels: int):
    """Load PEFT adapter, merge into base model, save as standard transformer."""
    print(f"\nBase model: {base_model_name} | Adapter: {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=num_labels)

    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    peft_model.eval()

    test_input = tokenizer("test input", return_tensors="pt")
    with torch.no_grad():
        logits_before = peft_model(**test_input).logits

    merged_model = peft_model.merge_and_unload()

    with torch.no_grad():
        logits_after = merged_model(**test_input).logits

    diff = (logits_before - logits_after).abs().max().item()
    print(f"  Merge validation — max logit diff: {diff:.6f} (should be ~0)")

    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"  Saved to {merged_dir}/")


def main():
    label_classes = json.loads((ARTIFACT_DIR / "label_classes.json").read_text(encoding="utf-8"))
    num_labels = len(label_classes)
    print(f"Labels ({num_labels}): {label_classes}")

    merge_and_save(DISTIL_ADAPTER_DIR, DISTIL_BASE, DISTIL_MERGED_DIR, num_labels)
    merge_and_save(ALBERT_ADAPTER_DIR, ALBERT_BASE, ALBERT_MERGED_DIR, num_labels)

    print("\n✅ All models merged! Run: streamlit run app.py")


if __name__ == "__main__":
    main()
