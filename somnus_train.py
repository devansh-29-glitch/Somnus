# somnus_train.py
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier

from mne.datasets.sleep_physionet import age
from somnus_features import load_psg_hyp
from somnus_utils import INV_STAGE_MAP

ASSETS = Path("assets")
ASSETS.mkdir(exist_ok=True)

def load_subset():
    # Load same subset we downloaded: subjects [0,1], recording 1
    records = age.fetch_data(subjects=[0, 1], recording=[1])
    X_list, y_list = [], []
    for psg_path, hyp_path in records:
        X, y = load_psg_hyp(psg_path, hyp_path)
        # Remove epochs labeled outside our mapping (just in case)
        mask = (y >= 0) & (y <= 4)
        X_list.append(X[mask])
        y_list.append(y[mask])
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y

def plot_confusion(y_true, y_pred, outpath):
    labels = [INV_STAGE_MAP[i] for i in sorted(np.unique(y_true))]
    cm = confusion_matrix(y_true, y_pred, labels=sorted(np.unique(y_true)))
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Somnus — Sleep Stage Confusion Matrix")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_class_hist(y, outpath):
    labels = [INV_STAGE_MAP[i] for i in sorted(np.unique(y))]
    counts = [np.sum(y==i) for i in sorted(np.unique(y))]
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x=labels, y=counts, ax=ax)
    ax.set_title("Class Distribution (Epochs)")
    ax.set_xlabel("Stage"); ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    print("Loading data...")
    X, y = load_subset()
    print("Feature matrix:", X.shape, "Labels:", y.shape)

    # Save class hist
    plot_class_hist(y, ASSETS / "class_distribution.png")

    # Train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced_subsample')
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)

    print(f"\nSomnus accuracy: {acc*100:.2f}%  |  Balanced Acc: {bacc*100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=[INV_STAGE_MAP[i] for i in sorted(np.unique(y))]))

    # Confusion matrix
    plot_confusion(y_test, y_pred, ASSETS / "confusion_matrix.png")

    # Save model if you want later (joblib)
    # from joblib import dump
    # dump(clf, "somnus_model.joblib")
    print("\nSaved figures in assets/: class_distribution.png, confusion_matrix.png")
