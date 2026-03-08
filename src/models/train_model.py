import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from src.features.feature_engineering import create_vectorizer
from src.utils.helpers import save_object

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "preprocessed_toxicity_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "trained_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

def train_model():
    data = pd.read_csv(DATA_PATH)
    X = data["processed_text"]
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = create_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    save_object(model, MODEL_PATH)
    save_object(vectorizer, VECTORIZER_PATH)
    print("Model training complete.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")

if __name__ == "__main__":
    train_model()