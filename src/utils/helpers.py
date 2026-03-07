import joblib
def save_object(obj, path):
    """save python object to disk"""
    joblib.dump(obj, path)


def load_object(path):
    """load python object from disk"""
    return joblib.load(path)