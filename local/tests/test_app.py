import pytest
from fastapi.testclient import TestClient
from app.app import app

client = TestClient(app)


# Test root endpoint
def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Custom Sentiment API is running!"}


# Test valid prediction
def test_valid_prediction():
    response = client.post(
        "/predict/", json={"text": "This book is absolutely fantastic!"}
    )
    assert response.status_code == 200
    assert "sentiment" in response.json()
    assert response.json()["sentiment"] in ["positive", "neutral", "negative"]


# Test empty text prediction
def test_empty_text_prediction():
    response = client.post("/predict/", json={"text": ""})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "neutral"


# Test missing text field
def test_missing_text_field():
    response = client.post("/predict/", json={})
    assert response.status_code == 422


# Test invalid JSON format
def test_invalid_json_format():
    response = client.post("/predict/", data="invalid_json")
    assert response.status_code == 422


# Test single-word input
def test_single_word_text():
    response = client.post("/predict/", json={"text": "Boring"})
    assert response.status_code == 200
    assert "sentiment" in response.json()
    assert response.json()["sentiment"] in ["positive", "neutral", "negative"]


# Test long text input
def test_long_text_prediction():
    long_text = "Amazing! " * 1000  # Long review
    response = client.post("/predict/", json={"text": long_text})
    assert response.status_code == 200
    assert "sentiment" in response.json()
    assert response.json()["sentiment"] in ["positive", "neutral", "negative"]
