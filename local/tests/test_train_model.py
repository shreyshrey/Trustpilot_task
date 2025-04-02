import os
import json
import pytest
import numpy as np
from unittest.mock import patch
from model.train_model import (
    map_rating_to_label,
    load_and_preprocess_data,
    train_and_save_model
)

# Test constants
TEST_FILE_PATH = "tests/test_data.jsonl"
MODEL_OUTPUT_PATH = "tests/test_model.pth"
VECTORIZER_PATH = "tests/test_vectorizer.npy"

# Sample data for testing
TEST_DATA = [
    {"rating": 5.0, "text": "This book is amazing. I could read this over and over. Love this book"},
    {"rating": 1.0, "text": "Horrible book. Waste of time."},
    {"rating": 3.0, "text": "It's an okay book."},
]

@pytest.fixture
def create_test_data():
    """Create a temporary JSONL file for testing."""
    with open(TEST_FILE_PATH, "w") as f:
        for entry in TEST_DATA:
            f.write(json.dumps(entry) + "\n")
    yield
    os.remove(TEST_FILE_PATH)

def test_map_rating_to_label():
    """Test the function for mapping ratings to sentiment labels."""
    assert map_rating_to_label(5.0) == "positive"
    assert map_rating_to_label(3.0) == "neutral"
    assert map_rating_to_label(1.0) == "negative"

@pytest.mark.usefixtures("create_test_data")
def test_load_and_preprocess_data():
    """Test loading and preprocessing data."""
    try:
        DATA_PATH = TEST_FILE_PATH  # Define DATA_PATH for testing
        sentences, labels = load_and_preprocess_data(DATA_PATH)
        assert len(sentences) > 0, "No sentences loaded!"
        assert len(labels) > 0, "No labels assigned!"

        for sentence, label in zip(sentences, labels):
            assert sentence.strip() != "", "Empty sentence found: {0}".format(sentence)
            assert label in ["negative", "neutral", "positive"], "Invalid label found: {0}".format(label)
        print("Data loading and preprocessing test passed! - {0} sentences with labels.".format(len(sentences)))
    except Exception as e:
        pytest.fail(f"Error in data loading or preprocessing: {e}")

    
def test_train_and_save_model(create_test_data):
    """Test model training and saving."""
    sentences, labels = load_and_preprocess_data(TEST_FILE_PATH)

    # Train the model and save it
    train_and_save_model(sentences, labels, MODEL_OUTPUT_PATH, VECTORIZER_PATH)

    # Ensure model and vectorizer are saved
    assert os.path.exists(MODEL_OUTPUT_PATH)
    assert os.path.exists(VECTORIZER_PATH)

    # Ensure the model is a PyTorch file and vectorizer is a numpy file
    assert MODEL_OUTPUT_PATH.endswith(".pth")
    assert VECTORIZER_PATH.endswith(".npy")

def test_edge_case_empty_data():
    """Test edge case: empty data"""
    empty_sentences = []
    empty_labels = []
    
    # Expecting an error because the data is empty
    with pytest.raises(ValueError):
        train_and_save_model(empty_sentences, empty_labels, MODEL_OUTPUT_PATH, VECTORIZER_PATH)

def test_invalid_file_path():
    """Test loading data from an invalid file path."""
    invalid_path = "invalid_path.jsonl"
    with pytest.raises(FileNotFoundError):
        load_and_preprocess_data(invalid_path)

def test_malformed_json_data():
    """Test loading data with malformed JSON."""
    malformed_data_path = "tests/malformed_data.jsonl"
    with open(malformed_data_path, "w") as f:
        f.write("{invalid_json}\n")  # Write malformed JSON
    try:
        with pytest.raises(json.JSONDecodeError):
            load_and_preprocess_data(malformed_data_path)
    finally:
        os.remove(malformed_data_path)

