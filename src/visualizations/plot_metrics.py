import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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