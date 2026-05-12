import numpy as np
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
# 🔹 EVALUATION COMPLETA RETE NEURALE
# =========================================================

def run_full_evaluation_nn(model, X_test, y_test, title="Neural Network Test"):

    # =========================
    # PREDIZIONI
    # =========================

    y_pred = model.predict(X_test)

    # Probabilità (serve per AUC)
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]
    else:
        y_probs = None

    # =========================
    # METRICHE
    # =========================

    metrics = {

        "Accuracy":
            accuracy_score(y_test, y_pred),

        "Precision":
            precision_score(y_test, y_pred),

        "Recall":
            recall_score(y_test, y_pred),

        "F1-Score":
            f1_score(y_test, y_pred),

        "AUC-ROC":
            roc_auc_score(y_test, y_probs) if y_probs is not None else None
    }

    # =========================
    # REPORT
    # =========================

    print(f"\n{'=' * 60}")
    print(f" NEURAL NETWORK EVALUATION: {title}")
    print(f"{'=' * 60}")

    for k, v in metrics.items():
        print(f"{k:10}: {v:.4f}" if v is not None else f"{k:10}: N/A")

    print("\n Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # =========================
    # CONFUSION MATRIX
    # =========================

    cm = confusion_matrix(y_test, y_pred)

    plot_confusion_matrix(cm, title)

    return metrics, cm


# =========================================================
# 🔹 CONFUSION MATRIX PLOT
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
# 🔹 SUMMARY K-FOLD NEURAL NETWORK
# =========================================================

def print_kfold_summary_nn(all_metrics, all_confusion_matrices):

    print(f"\n{'=' * 70}")
    print(" NEURAL NETWORK - K-FOLD SUMMARY")
    print(f"{'=' * 70}")

    avg_metrics = {}

    for k in all_metrics[0]:

        avg_metrics[k] = np.mean(
            [m[k] for m in all_metrics]
        )

    std_acc = np.std(
        [m["Accuracy"] for m in all_metrics]
    )

    for k, v in avg_metrics.items():
        print(f"{k:10}: {v:.4f}")

    print(f"\nStd Accuracy: {std_acc:.4f}")

    avg_cm = np.mean(all_confusion_matrices, axis=0)

    print(f"\n Confusion Matrix Media:\n")
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

    plt.title("Confusion Matrix Media - NN K-Fold")

    plt.show()

def plot_loss_curve(model):
    """
    Plotta training loss e validation loss (approssimata).
    """

    plt.figure(figsize=(8, 5))

    # 🔵 Training loss (reale)
    plt.plot(
        model.loss_curve_,
        label="Training Loss"
    )

    # 🟠 Validation score (se early stopping attivo)
    if hasattr(model, "validation_scores_"):
        plt.plot(
            model.validation_scores_,
            label="Validation Score (proxy)"
        )

    plt.title("Loss Curve - Neural Network")
    plt.xlabel("Epochs")
    plt.ylabel("Loss / Score")
    plt.legend()
    plt.grid(True)

    plt.show()