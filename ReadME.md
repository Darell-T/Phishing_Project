# Phishing Email Detection with BERT

This project uses a fine-tuned BERT model to classify emails as **phishing** or **legitimate** emails.

## Features

- Cleans and preprocesses email text
- Trains a BERT model on labeled email data
- Evaluates model performance
- Predicts on new emails or batches

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers
- scikit-learn
- pandas

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/phishing-email-bert.git
    cd phishing-email-bert
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Place your emails dataset in the project directory.

### Training

To train and save the model:
```bash
python data.py
```

### Prediction

To predict on new emails or a batch:
```bash
python predict.py
```
Edit `predict.py` to change the input email or batch size.

## File Structure

- `data.py` — Training and evaluation script
- `predict.py` — Prediction script for new emails
- `phishing_email.csv` — Your dataset (not included in repo)
- `saved_model/` — Saved model and tokenizer (created after training)

## Example

```
Prediction: Phishing (Confidence: 98.53%)
```




