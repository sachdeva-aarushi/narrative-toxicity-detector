import pytest
from src.models.predict import predict_text
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_clean_sentence_prediction():
    result = predict_text("I really like this project")
    assert result["prediction"] in ["neutral", "toxic"]


def test_toxic_sentence_prediction():
    result = predict_text("You are the dumbest person ever")
    assert result["prediction"] == "toxic"


def test_empty_input():
    with pytest.raises(ValueError):
        predict_text("")


def test_long_input():
    long_text = "A" * 12000
    with pytest.raises(ValueError):
        predict_text(long_text)