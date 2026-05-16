import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)


def run_full_evaluation(model, X_test, y_test, title="Valutazione Modello", verbose=True):
    """
    Calcola le metriche, genera la Matrice di Confusione e la Curva ROC.
    Restituisce il dizionario delle metriche e la matrice di confusione.
    """
    # 1. Generazione Predizioni e Probabilità
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    # Safeguard: Convertiamo in booleani per sicurezza (CatBoost a volte restituisce stringhe 'True'/'False')
    y_test = y_test.astype(bool)
    if isinstance(y_pred[0], str):
        y_pred = np.array([True if str(x).lower() == 'true' else False for x in y_pred])
    else:
        y_pred = y_pred.astype(bool)

    # 2. Calcolo Metriche
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs)

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc
    }

    # 3. Stampe e Grafici (solo se verbose=True, es. nell'Holdout)
    if verbose:
        print(f"\n{'-' * 50}")
        print(f" 📊 REPORT METRICHE: {title.upper()}")
        print(f"{'-' * 50}")
        print(f"   🎯 Accuracy  : {acc:.4f}")
        print(f"   🔎 Precision : {prec:.4f}")
        print(f"   🎣 Recall    : {rec:.4f}")
        print(f"   ⚖️ F1-Score  : {f1:.4f}")
        print(f"   📈 ROC-AUC   : {roc_auc:.4f}")
        print(f"{'-' * 50}\n")

        # --- GRAFICI: MATRICE DI CONFUSIONE + CURVA ROC ---
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"{title}", fontsize=16)

        # Matrice di Confusione
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Matrice di Confusione')
        axes[0].set_xlabel('Predetto')
        axes[0].set_ylabel('Reale')

        # Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        axes[1].plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", color='darkorange', lw=2)
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_title('Curva ROC')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].legend(loc="lower right")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return metrics, cm


def print_kfold_summary(fold_metrics, fold_confusion_matrices):
    """
    Stampa la media delle metriche dei singoli Fold e la Matrice Totale per i modelli dei tuoi compagni.
    """
    print("\n" + "=" * 50)
    print(" 🏆 SINTESI RISULTATI K-FOLD (Medie)")
    print("=" * 50)

    avg_acc = np.mean([m['accuracy'] for m in fold_metrics])
    avg_prec = np.mean([m['precision'] for m in fold_metrics])
    avg_rec = np.mean([m['recall'] for m in fold_metrics])
    avg_f1 = np.mean([m['f1'] for m in fold_metrics])

    # Aggiungiamo il calcolo del ROC AUC medio
    if 'roc_auc' in fold_metrics[0]:
        avg_roc = np.mean([m['roc_auc'] for m in fold_metrics])
    else:
        avg_roc = None

    print(f"   🎯 Accuracy Media  : {avg_acc:.4f}")
    print(f"   🔎 Precision Media : {avg_prec:.4f}")
    print(f"   🎣 Recall Media    : {avg_rec:.4f}")
    print(f"   ⚖️ F1-Score Medio  : {avg_f1:.4f}")
    if avg_roc is not None:
        print(f"   📈 ROC-AUC Medio   : {avg_roc:.4f}")
    print("=" * 50 + "\n")

    # Matrice di confusione cumulativa
    total_cm = np.sum(fold_confusion_matrices, axis=0)

    plt.figure(figsize=(7, 5))
    sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matrice di Confusione Cumulativa (Tutti i Fold)")
    plt.xlabel("Predetto")
    plt.ylabel("Reale")
    plt.show()