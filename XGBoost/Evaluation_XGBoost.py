import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


def run_full_evaluation(model, X_test, y_test, title="Test Set"):
    """
    Esegue una valutazione completa e stampa i risultati.
    """
    # Predizioni secche e probabilità
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    # Calcolo metriche richieste
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_probs)
    }

    print(f"\n --- REPORT VALUTAZIONE: {title} ---")
    for name, value in metrics.items():
        print(f"{name:10}: {value:.4f}")

    # Matrice di Confusione
    plot_confusion_matrix(y_test, y_pred, title)

    return metrics


def plot_confusion_matrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non Trasp.', 'Trasp.'],
                yticklabels=['Non Trasp.', 'Trasp.'])
    plt.xlabel('Predetto')
    plt.ylabel('Reale')
    plt.title(f'Matrice di Confusione - {title}')
    plt.show()