import json
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import os
from collections import Counter

# Setting up logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

nltk.download("punkt")

# File paths
FILE_PATH = "./data/Books_10k.jsonl"
MODEL_OUTPUT_PATH = "./model_output/sentiment_model.pth"
VECTORIZER_PATH = "./model_output/vectorizer.npy"
LABEL_ENCODER_PATH = "./model_output/label_encoder.npy"

os.makedirs("./model_output", exist_ok=True)

# Map ratings to sentiment labels
def map_rating_to_label(rating):
    """
    Map numerical star ratings to sentiment labels.
    """
    sentiment = "neutral"
    if rating >= 4.0:
        return "positive" 
    elif rating <= 2.0:
        return "negative"
    return sentiment


# Load and preprocess data
def load_and_preprocess_data(file_path):
    """
    Load JSONL data, split reviews into sentences, and assign labels.
    """
    logging.info(f"Loading data from {file_path}")
    data = []
    with open(file_path, "r") as f:
        for line in tqdm(f, desc="Reading data", unit="lines"):
            data.append(json.loads(line))

    sentences, labels = [], []
    logging.info("Splitting reviews into sentences and assigning labels")

    for review in tqdm(data, desc="Processing reviews", unit="reviews"):
        review_text = review["text"]
        rating = review["rating"]
        label = map_rating_to_label(rating)

        # Split review into sentences and assign labels
        for sentence in sent_tokenize(review_text):
            sentences.append(sentence)
            labels.append(label)
        
    logging.info(f"Loaded and preprocessed {len(sentences)} sentences")
    return sentences, labels


class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=3):  
        super(SentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Helps prevent overfitting
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)  # Forces the model to predict probabilities

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)  # Ensure the output is interpretable as probabilities


def save_label_encoder(label_encoder, path):
    """Save the label encoder to a file."""
    np.save(path, label_encoder.classes_)
    logging.info(f"Label encoder saved to {path}")

def load_label_encoder(path):
    """Load the label encoder from a file."""
    classes = np.load(path, allow_pickle=True)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = classes
    logging.info(f"Label encoder loaded from {path}")
    return label_encoder

def train_model(model, criterion, optimizer, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, epochs=10):
    """Train the model and evaluate on the validation set."""
    logging.info("Starting training...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)

        logging.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Training Loss: {loss.item():.4f} | "
            f"Validation Loss: {val_loss.item():.4f}"
        )

# Update train_and_save_model to include label encoder saving
def train_and_save_model(sentences, labels, model_output_path, vectorizer_path, label_encoder_path):
    """
    Train a sentiment classifier and save the model, vectorizer, and label encoder.
    """
    logging.info("Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(sentences).toarray()

    # Encode labels to integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    logging.info("Encoded labels: %s", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

    # Split data into train and test sets
    logging.info("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # Define model, loss, and optimizer
    input_dim = X_train.shape[1]
    model = SentimentClassifier(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    train_model(model, criterion, optimizer, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, epochs=10)

    # Save model, vectorizer, and label encoder
    torch.save(model.state_dict(), model_output_path)
    np.save(vectorizer_path, vectorizer)
    save_label_encoder(label_encoder, label_encoder_path)

    logging.info(f"Model saved to {model_output_path}")
    logging.info(f"Vectorizer saved to {vectorizer_path}")
    logging.info(f"Label encoder saved to {label_encoder_path}")


if __name__ == "__main__":
    sentences, labels = load_and_preprocess_data(FILE_PATH)
    train_and_save_model(sentences, labels, MODEL_OUTPUT_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH)
