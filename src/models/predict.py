import os
from src.utils.helpers import load_object
from src.data.preprocessing import clean_text
from src.features.feature_engineering import transform_text
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_PATH = os.path.join(BASE_DIR, "models", "trained_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")
def load_artifacts():
    model = load_object(MODEL_PATH)
    vectorizer = load_object(VECTORIZER_PATH)
    return model, vectorizer

def predict_text(text):
    model, vectorizer = load_artifacts()
    cleaned_text = clean_text(text)
    features = transform_text(vectorizer, [cleaned_text])
    prediction = model.predict(features)[0]
    return "toxic" if prediction == 1 else "neutral"