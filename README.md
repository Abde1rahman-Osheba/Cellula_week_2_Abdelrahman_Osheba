# Cellula Week 2 — Text & Image Classification

A **Streamlit** application for toxic content classification using a hybrid ensemble of:

- **BLIP-1** for image captioning (Hugging Face `Salesforce/blip-image-captioning-large`)
- **Fine-tuned DistilBERT with LoRA** for text classification
- **Fine-tuned ALBERT with LoRA** for text classification
- **Bidirectional LSTM** for text classification
- **Logistic Regression meta-model** combining all three classifiers

## Project Structure

| File | Description |
|------|-------------|
| `app.py` | Streamlit application (main entry point) |
| `imagecaption.py` | Image captioning module using BLIP-1 |
| `text_classifier.py` | Hybrid ensemble text classifier |
| `database.py` | CSV-based database for storing classification results |
| `requirements.txt` | Python dependencies |
| `cellula toxic data.csv` | Training dataset |

## Features

- **Classify Text** — Enter text and get toxicity classification
- **Classify Image** — Upload an image, auto-generate caption, classify it
- **View Database** — Browse all past classification results stored in CSV

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

This app is ready for **Streamlit Community Cloud**:

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set the main file to `app.py`

## Author

**Abdelrahman Osheba**
