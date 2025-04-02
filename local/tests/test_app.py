import pytest
from fastapi.testclient import TestClient
from app.app import app

# Initialize the FastAPI test client
client = TestClient(app)

# Sample review data
valid_review = {"text": "This book is absolutely amazing, I loved it!"}
invalid_review_empty = {"text": ""}
invalid_review_no_text = {}

def test_root_endpoint():
    """Test root endpoint to check if the API is running."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Custom Sentiment API is running!"}



def test_predict_valid_review():
    """Test prediction with valid input."""
    response = client.post("/predict/", json=valid_review)
    assert response.status_code == 200
    assert "sentiment" in response.json()
    assert response.json()["sentiment"] in ["positive", "neutral", "negative"]

def test_predict_invalid_missing_text():
    """Test prediction with missing review text."""
    response = client.post("/predict/", json=invalid_review_no_text)
    assert response.status_code == 422
    assert "detail" in response.json()

def test_predict_invalid_json_format():
    """Test prediction with invalid JSON format."""
    response = client.post("/predict/", data="invalid_json")
    assert response.status_code == 422
    assert "detail" in response.json()

def test_predict_edge_case_non_string_input():
    """Test prediction with non-string input (numeric)."""
    response = client.post("/predict/", json={"text": 12345})
    assert response.status_code == 422
    assert "detail" in response.json()
