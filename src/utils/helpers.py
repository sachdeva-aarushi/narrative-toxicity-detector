import joblib
def save_object(obj, path):
    """save python object to disk"""
    joblib.dump(obj, path)


def load_object(path):
    """load python object from disk"""
    return joblib.load(path)

import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("toxicity_detector")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler("logs/inference.log",maxBytes=5_000_000,backupCount=3)

formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)