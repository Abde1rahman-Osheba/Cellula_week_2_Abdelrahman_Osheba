"""
database.py
-----------
CSV-based database module for storing and retrieving classification records.
The CSV file is automatically created on first use and updated whenever the
user submits a text input or an image caption for classification.
"""

import os
from datetime import datetime

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = "classifications_db.csv"
COLUMNS = ["timestamp", "input_type", "input_text", "classification_label", "confidence"]


def _ensure_db_exists():
    """Create the CSV database file with headers if it does not exist."""
    if not os.path.exists(DB_PATH):
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(DB_PATH, index=False)


def save_record(
    input_type: str,
    input_text: str,
    classification_label: str,
    confidence: float,
) -> None:
    """
    Append a new classification record to the CSV database.

    Parameters
    ----------
    input_type : str
        Either "text" or "image".
    input_text : str
        The raw user text or the generated image caption.
    classification_label : str
        The predicted classification label.
    confidence : float
        The confidence score of the prediction.
    """
    _ensure_db_exists()
    new_row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "input_type": input_type,
                "input_text": input_text,
                "classification_label": classification_label,
                "confidence": round(confidence, 4),
            }
        ]
    )
    new_row.to_csv(DB_PATH, mode="a", header=False, index=False)


def load_records() -> pd.DataFrame:
    """
    Load all stored classification records from the CSV database.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing all records, or an empty DataFrame if the
        database does not exist yet.
    """
    _ensure_db_exists()
    return pd.read_csv(DB_PATH)
