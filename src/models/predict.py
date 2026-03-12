import os
from src.utils.helpers import load_object
from src.data.preprocessing import clean_text
from src.features.feature_engineering import transform_text
import yaml
from src.utils.helpers import logger
import time
from datetime import datetime
import json
from pathlib import Path
start_time = time.time()
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
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    if text.strip() == "":
        raise ValueError("Input text cannot be empty")

    if len(text) > 10000:
        raise ValueError("Input text exceeds maximum allowed length")
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
latency = (time.time() - start_time) * 1000
log_data = {
    "timestamp": datetime.utcnow().isoformat(),
    "input_length": len(text),
    "toxicity_score": toxicity_score,
    "prediction": prediction,
    "latency_ms": round(latency,2)
}
logger.info(json.dumps(log_data))