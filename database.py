import os
from datetime import datetime

import pandas as pd

DB_PATH = "classifications_db.csv"
COLUMNS = ["timestamp", "input_type", "input_text", "classification_label", "confidence"]


def _ensure_db_exists():
    if not os.path.exists(DB_PATH):
        pd.DataFrame(columns=COLUMNS).to_csv(DB_PATH, index=False)


def save_record(input_type: str, input_text: str, classification_label: str, confidence: float) -> None:
    """Append a classification record to the CSV database."""
    _ensure_db_exists()
    new_row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_type": input_type,
        "input_text": input_text,
        "classification_label": classification_label,
        "confidence": round(confidence, 4),
    }])
    new_row.to_csv(DB_PATH, mode="a", header=False, index=False)


def load_records() -> pd.DataFrame:
    """Load all classification records from the CSV database."""
    _ensure_db_exists()
    return pd.read_csv(DB_PATH)
