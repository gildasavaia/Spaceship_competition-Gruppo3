import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def run_full_evaluation(model, X_test, y_test, title="Test Set", verbose=True):

    
    # Predizioni

    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]


    # Metriche

    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1-Score":  f1_score(y_test, y_pred),
        "AUC-ROC":   roc_auc_score(y_test, y_probs)
    }


    # Confusion Matrix

    cm = confusion_matrix(y_test, y_pred)


    # REPORT E PLOT (Solo se verbose è True)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"📊 REPORT VALUTAZIONE: {title}")
        print(f"{'=' * 60}")

        for metric_name, metric_value in metrics.items():
            print(f"{metric_name:10}: {metric_value:.4f}")

        plot_confusion_matrix(cm, title)

    return metrics, cm


def plot_confusion_matrix(
        cm,
        title
):

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        cm,

        annot=True,

        fmt='d',

        cmap='Blues',

        xticklabels=[
            'Non Trasp.',
            'Trasp.'
        ],

        yticklabels=[
            'Non Trasp.',
            'Trasp.'
        ]
    )

    plt.xlabel('Predetto')

    plt.ylabel('Reale')

    plt.title(
        f'Matrice di Confusione - {title}'
    )

    plt.show()


def print_kfold_summary(
        all_metrics,
        all_confusion_matrices
):

    print(f"\n{'=' * 70}")
    print("🏆 RISULTATI MEDI FINALI K-FOLD CATBOOST")
    print(f"{'=' * 70}")

    avg_metrics = {}

    for metric_name in all_metrics[0]:

        avg_metrics[metric_name] = np.mean(
            [
                m[metric_name]
                for m in all_metrics
            ]
        )

    std_accuracy = np.std(
        [
            m["Accuracy"]
            for m in all_metrics
        ]
    )

    for metric_name, metric_value in avg_metrics.items():

        print(
            f"{metric_name:10}: "
            f"{metric_value:.4f}"
        )

    print(
        f"\nStd Accuracy: "
        f"{std_accuracy:.4f}"
    )

    avg_cm = np.mean(
        all_confusion_matrices,
        axis=0
    )

    print(f"\n{'=' * 70}")
    print("📌 MATRICE DI CONFUSIONE MEDIA")
    print(f"{'=' * 70}")

    print(avg_cm)

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        avg_cm,

        annot=True,

        fmt='.2f',

        cmap='Blues',

        xticklabels=[
            'Non Trasp.',
            'Trasp.'
        ],

        yticklabels=[
            'Non Trasp.',
            'Trasp.'
        ]
    )

    plt.xlabel('Predetto')

    plt.ylabel('Reale')

    plt.title(
        'Matrice di Confusione Media K-Fold CATBOOST'
    )

    plt.show()