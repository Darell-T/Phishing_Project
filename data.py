import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import sys
import transformers

# --- Environment Info ---
print(f"Python version: {sys.version}")
print(f"Transformers version: {transformers.__version__}")

# --- Load and Inspect Data ---
df = pd.read_csv("phishing_email.csv")
print("\nMissing values per column:\n", df.isnull().sum())
print("\nClass distribution:\n", df['label'].value_counts())

# --- Text Cleaning Function ---
def clean_text(text):
    """Lowercase, remove links, emails, numbers, punctuation, and extra spaces."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)         # Remove URLs
    text = re.sub(r"\S+@\S+", "", text)         # Remove email addresses
    text = re.sub(r"\d+", "", text)             # Remove numbers
    text = re.sub(r"[^\w\s]", "", text)         # Remove punctuation
    text = re.sub(r"\s+", " ", text)            # Remove extra spaces
    return text.strip()

# --- Clean the Text Data ---
df['clean_text'] = df['text_combined'].apply(clean_text)
print("\nSample cleaned data:\n", df[['text_combined', 'clean_text', 'label']].head())

# --- Train/Test Split ---
X = df['clean_text'].tolist()
y = df['label'].tolist()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Tokenization ---
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

# --- Dataset Wrapper ---
class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = EmailDataset(train_encodings, y_train)
test_dataset = EmailDataset(test_encodings, y_test)

# --- Model Setup ---
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    logging_steps=10,
    disable_tqdm=False,
    no_cuda=False  # Use GPU if available
)

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# --- Train the Model ---
trainer.train()

# --- Evaluate the Model ---
preds = trainer.predict(test_dataset)
y_pred = preds.predictions.argmax(axis=1)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

# --- Save the Model and Tokenizer ---
output_dir = "./saved_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\nModel and tokenizer saved to {output_dir}")

