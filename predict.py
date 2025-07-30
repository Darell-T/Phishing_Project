import torch
from transformers import BertTokenizer, BertForSequenceClassification
from data import clean_text 
import pandas as pd
from sklearn.metrics import accuracy_score

# Load the saved model and tokenizer
model_path = "./saved_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Move to GPU 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()  # Set model to evaluation mode for testing

def predict_email(text):
    # Preprocess the text
    cleaned_text = clean_text(text)
    
    # Tokenize
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=1)
        confidence = probabilities[0][prediction].item()
    
    return {
        "prediction": "Phishing" if prediction.item() == 1 else "Legitimate",
        "confidence": confidence,
        "pred_label": prediction.item()
    }

# Email tester
if __name__ == "__main__":
    df = pd.read_csv("phishing_email.csv")
    sample_df = df.head(16000)  # Use the first 16,000 emails

    y_true = []
    y_pred = []

    for idx, row in sample_df.iterrows():
        email_text = row['text_combined']
        true_label = row['label']
        result = predict_email(email_text)
        y_true.append(true_label)
        y_pred.append(result['pred_label'])

        if idx < 10:  # Print only the first 10 predictions
            print(f"Email #{idx+1} | True: {true_label} | Predicted: {result['pred_label']} | {result['prediction']} ({result['confidence']:.2%})")

    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy on first 16,000 emails: {acc:.2%}")

    #Find another dataset to test the model