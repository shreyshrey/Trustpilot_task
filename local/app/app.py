from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from model.train_model import SentimentClassifier
import logging


# Setting up logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

# Define the FastAPI app
app = FastAPI()

MODEL_PATH = "./model_ouput/sentiment_model.pth"
VECTORIZER_PATH = "./model_ouput/vectorizer.npy"
# Define label mapping
LABELS = ["negative", "neutral", "positive"]

class ReviewRequest(BaseModel):
    text: str

try:

    # Load vectorizer and model
    vectorizer = np.load(VECTORIZER_PATH, allow_pickle=True).item()
    input_dim = len(vectorizer.get_feature_names_out())
    model = SentimentClassifier(input_dim)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    logging.info("Model loaded successfully!")
except Exception as err:
    logging.error("Error loading model {0}".format(err))
    raise err



# Prediction endpoint
@app.post("/predict/")
async def predict(request: ReviewRequest):
    """
    Predict sentiment for the given review text.
    """
    logging.info("🔎 Received prediction request: {0}...".format(request.text[:50]))

    text = request.text
    try:

        # Vectorize input
        X_input = vectorizer.transform([text]).toarray()
        X_tensor = torch.tensor(X_input, dtype=torch.float32)

        # Predict sentiment
        with torch.no_grad():
            outputs = model(X_tensor)
            prediction = torch.argmax(outputs, dim=1).item()

        sentiment = LABELS[prediction]
        logging.info("🚀 Prediction: {0}".format(sentiment))
        return {"sentiment": sentiment}
    except Exception as err:
        logging.error("Error predicting sentiment: {0}".format(err))
        raise HTTPException(status_code=500, detail="Error predicting sentiment")



# Root endpoint for health check
@app.get("/")
def read_root():
    """
    Root endpoint to check API status.
    """
    logging.info(" Health check requested")
    return {"message": "Custom Sentiment API is running!"}
