import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# =========================================================
# 🔹 EVALUATION SINGOLO MODELLO
# =========================================================

def run_full_evaluation_nn(model, X_test, y_test, title="Neural Network", verbose=True):

    model.eval()

    # =========================
    # TENSOR INPUT
    # =========================
    X_tensor = torch.tensor(
        X_test.values if hasattr(X_test, "values") else X_test,
        dtype=torch.float32
    )

    with torch.no_grad():
        outputs = model(X_tensor).squeeze()
        y_probs = outputs.numpy()
        y_pred = (outputs > 0.5).int().numpy()

    # =========================
    # METRICHE
    # =========================
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_probs)
    }

    # =========================
    # CONFUSION MATRIX
    # =========================
    cm = confusion_matrix(y_test, y_pred)

    # =========================
    # PRINT REPORT (Solo se verbose è True)
    # =========================
    if verbose:
        print(f"\n{'=' * 60}")
        print(f" NEURAL NETWORK - {title}")
        print(f"{'=' * 60}")

        for k, v in metrics.items():
            print(f"{k:10}: {v:.4f}")

        print("\n Classification Report:\n")
        print(classification_report(y_test, y_pred))

        plot_confusion_matrix(cm, title)

    return metrics, cm
# =========================================================
# 🔹 CONFUSION MATRIX
# =========================================================

def plot_confusion_matrix(cm, title):

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=["False", "True"],
        yticklabels=["False", "True"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {title}")
    plt.show()


# =========================================================
# 🔹 K-FOLD SUMMARY
# =========================================================

def print_kfold_summary_nn(all_metrics, all_cms):

    print(f"\n{'=' * 70}")
    print(" NEURAL NETWORK - K-FOLD FINAL SUMMARY")
    print(f"{'=' * 70}")

    avg_metrics = {}

    for k in all_metrics[0]:

        avg_metrics[k] = np.mean([m[k] for m in all_metrics])

    std_acc = np.std([m["Accuracy"] for m in all_metrics])

    for k, v in avg_metrics.items():
        print(f"{k:10}: {v:.4f}")

    print(f"\nStd Accuracy: {std_acc:.4f}")

    # =========================
    # CONFUSION MATRIX MEDIA
    # =========================

    avg_cm = np.mean(all_cms, axis=0)

    print("\n Confusion Matrix Media:\n")
    print(avg_cm)

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        avg_cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["False", "True"],
        yticklabels=["False", "True"]
    )

    plt.title("Confusion Matrix Media - NN")
    plt.show()


# =========================================================
# 🔹 LOSS CURVE
# =========================================================

def plot_loss_curve(model):

    plt.figure(figsize=(8, 5))

    if hasattr(model, "loss_history"):

        plt.plot(model.loss_history, label="Training Loss")

    plt.title("Loss Curve - Neural Network")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()