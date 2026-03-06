import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["Neutral","Toxic"],
    yticklabels=["Neutral","Toxic"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_class_distribution(labels, save_path=None):
    plt.figure(figsize=(6,5))
    sns.countplot(x=labels)
    plt.title("Balanced Dataset Class Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_top_toxic_words(model, vectorizer, top_n=20, save_path=None):

    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    top_indices = np.argsort(coefficients)[-top_n:]
    top_words = [feature_names[i] for i in top_indices]
    top_scores = coefficients[top_indices]
    plt.figure(figsize=(8,6))
    sns.barplot(x=top_scores, y=top_words)
    plt.title("Top Toxic Words Learned by Model")
    plt.xlabel("Importance Score")
    plt.ylabel("Word")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()