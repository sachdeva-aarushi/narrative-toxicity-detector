import os
from src.utils.helpers import load_object
from src.data.preprocessing import clean_text
from src.features.feature_engineering import transform_text
import yaml
from pathlib import Path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_PATH = os.path.join(BASE_DIR, "models", "trained_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

config_path = Path("config/thresholds.yaml")
with open(config_path, "r") as f:
    thresholds = yaml.safe_load(f)
DECISION_THRESHOLD = thresholds["decision_threshold"]

_model = load_object(MODEL_PATH)
_vectorizer = load_object(VECTORIZER_PATH)

def predict_text(text):
    cleaned_text = clean_text(text)
    features = transform_text(_vectorizer, [cleaned_text])
    probabilities = _model.predict_proba(features)[0]
    toxicity_score = float(probabilities[1])
    prediction = "toxic" if toxicity_score >= DECISION_THRESHOLD else "neutral"
    toxicity_percentage = round(toxicity_score * 100, 2)
    return {
        "prediction": prediction,
        "toxicity_score": toxicity_score,
        "toxicity_percentage": toxicity_percentage
    }