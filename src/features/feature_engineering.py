from sklearn.feature_extraction.text import TfidfVectorizer
def create_vectorizer():
    """TF-IDF vectorizer used for feature extraction."""
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=5, max_df=0.9)
    return vectorizer

def transform_text(vectorizer, text_data):
    """Transform text data into TF-IDF feature vectors."""
    return vectorizer.transform(text_data)