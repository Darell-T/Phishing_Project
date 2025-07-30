# Phishing Email Detection with BERT

This project uses a fine-tuned BERT model to classify emails as **phishing** or **legitimate**.

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

3. Place your dataset (`phishing_email.csv`) in the project directory.

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

## License

MIT License

---

### 3. **Initialize Git and Push to GitHub**

1. Initialize git:
    ```bash
    git init
    git add .
    git commit -m "Initial commit: Phishing Email Detection with BERT"
    ```

2. Create a new repo on [GitHub](https://github.com/new).

3. Link your local repo and push:
    ```bash
    git remote add origin https://github.com/yourusername/phishing-email-bert.git
    git branch -M main
    git push -u origin main
    ```

---

**Tip:**  
- Don’t commit large files or sensitive data (like your CSV).
- Add a `requirements.txt` with your dependencies:
    ```
    torch
    transformers
    scikit-learn
    pandas
    ```

Let me know if you want a sample `requirements.txt` or more README details!# Phishing Email Detection with BERT

This project uses a fine-tuned BERT model to classify emails as **phishing** or **legitimate**.

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

3. Place your dataset (`phishing_email.csv`) in the project directory.

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

## License

MIT License

---

### 3. **Initialize Git and Push to GitHub**

1. Initialize git:
    ```bash
    git init
    git add .
    git commit -m "Initial commit: Phishing Email Detection with BERT"
    ```

2. Create a new repo on [GitHub](https://github.com/new).

3. Link your local repo and push:
    ```bash
    git remote add origin https://github.com/yourusername/phishing-email-bert.git
    git branch -M main
    git push -u origin main
    ```

---

**Tip:**  
- Don’t commit large files or sensitive data (like your CSV).
- Add a `requirements.txt` with your dependencies:
    ```
    torch
    transformers
    scikit-learn
    pandas
    ```

Let me know if you want a sample `requirements.txt` or more README