import re
import string
def clean_text(text):
    """text cleaning function used across the pipeline."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text_series(text_series):
    """Apply cleaning to an entire pandas Series of text."""
    return text_series.apply(clean_text)