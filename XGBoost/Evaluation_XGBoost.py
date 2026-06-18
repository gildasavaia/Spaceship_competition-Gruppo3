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
    Esegue la valutazione completa delle performance del modello XGBoost.
    Calcola le metriche fondamentali di classificazione binaria, mostra la
    matrice di confusione e traccia la curva ROC.

    Parametri:
    ----------
    model : oggetto modello
        Il classificatore XGBoost addestrato da valutare.
    X_test : DataFrame o array-like
        La matrice delle feature del set di test.
    y_test : Series o array-like
        I target reali relativi al set di test.
    title : str, opzionale
        Il titolo identificativo da utilizzare nei report e nei grafici.
    verbose : bool, opzionale
        Se impostato su True, stampa i risultati a schermo e genera i grafici.

    Restituisce:
    -----------
    metrics : dict
        Dizionario contenente i valori di accuracy, precision, recall, f1 e roc_auc.
    cm : array
        La matrice di confusione calcolata.
    """
    # Generazione delle predizioni puntuali e delle probabilità stimate per la classe positiva
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    # Standardizzazione del target e delle predizioni in formato booleano per evitare anomalie di formato
    y_test = y_test.astype(bool)
    if isinstance(y_pred[0], str):
        y_pred = np.array([True if str(x).lower() == 'true' else False for x in y_pred])
    else:
        y_pred = y_pred.astype(bool)

    # Calcolo delle metriche di classificazione standard
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

    # Stampa testuale e visualizzazione dei grafici vincolata al flag verbose
    if verbose:
        print(f"\n{'-' * 50}")
        print(f" REPORT METRICHE: {title.upper()}")
        print(f"{'-' * 50}")
        print(f"   Accuracy  : {acc:.4f}")
        print(f"   Precision : {prec:.4f}")
        print(f"   Recall    : {rec:.4f}")
        print(f"   F1-Score  : {f1:.4f}")
        print(f"   ROC-AUC   : {roc_auc:.4f}")
        print(f"{'-' * 50}\n")

        # Configurazione dell'area di disegno per ospitare i due grafici affiancati
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"{title}", fontsize=16)

        # Matrice di Confusione (formato intero con heatmap blu)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Matrice di Confusione')
        axes[0].set_xlabel('Predetto')
        axes[0].set_ylabel('Reale')

        # Calcolo dei punti della curva ROC e plot grafico rispetto alla bisettrice casuale
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
    Calcola ed esibisce la media aritmetica delle performance ottenute
    nei singoli fold della Cross-Validation e mostra la matrice cumulativa.

    Parametri:
    ----------
    fold_metrics : list of dict
        Lista dei dizionari contenenti le metriche registrate per ogni fold.
    fold_confusion_matrices : list of arrays
        Lista delle matrici di confusione generate in ciascun fold.
    """
    print("\n" + "=" * 50)
    print(" SINTESI RISULTATI K-FOLD (Medie)")
    print("=" * 50)

    # Estrazione e calcolo della media aritmetica per ciascuna metrica monitorata
    avg_acc = np.mean([m['accuracy'] for m in fold_metrics])
    avg_prec = np.mean([m['precision'] for m in fold_metrics])
    avg_rec = np.mean([m['recall'] for m in fold_metrics])
    avg_f1 = np.mean([m['f1'] for m in fold_metrics])

    # Verifica preliminare dell'esistenza della metrica ROC-AUC prima del calcolo
    if 'roc_auc' in fold_metrics[0]:
        avg_roc = np.mean([m['roc_auc'] for m in fold_metrics])
    else:
        avg_roc = None

    print(f"   Accuracy Media  : {avg_acc:.4f}")
    print(f"   Precision Media : {avg_prec:.4f}")
    print(f"   Recall Media    : {avg_rec:.4f}")
    print(f"   F1-Score Medio  : {avg_f1:.4f}")
    if avg_roc is not None:
        print(f"   ROC-AUC Medio   : {avg_roc:.4f}")
    print("=" * 50 + "\n")

    # Somma algebrica delle singole matrici di ogni fold per ottenere la ripartizione totale
    total_cm = np.sum(fold_confusion_matrices, axis=0)

    # Rendering grafico della matrice di confusione aggregata totale
    plt.figure(figsize=(7, 5))
    sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matrice di Confusione Cumulativa (Tutti i Fold)")
    plt.xlabel("Predetto")
    plt.ylabel("Reale")
    plt.show()