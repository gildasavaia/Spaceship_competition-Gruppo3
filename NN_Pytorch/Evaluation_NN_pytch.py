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


def run_full_evaluation_nn(model, X_test, y_test, title="Neural Network", verbose=True):
    """
    Esegue la valutazione approfondita della rete neurale sul set di test.
    Converte i dati di input in tensori, ricava le probabilità e i valori
    predetti, calcola le metriche e visualizza la matrice di confusione.

    Parametri:
    ----------
    model : torch.nn.Module
        Il modello di rete neurale PyTorch già addestrato.
    X_test : DataFrame o array-like
        Le feature del set di test.
    y_test : Series o array-like
        I target reali del set di test.
    title : str, opzionale
        Il titolo da assegnare ai report e ai grafici.
    verbose : bool, opzionale
        Se impostato a True, stampa il report dettagliato e mostra la matrice.

    Restituisce:
    -----------
    metrics : dict
        Dizionario contenente i valori di Accuracy, Precision, Recall, F1-Score e AUC-ROC.
    cm : array
        La matrice di confusione bidimensionale.
    """
    # Imposta la rete in modalità valutazione per disattivare il Dropout e la Batch Normalization
    model.eval()

    # Conversione dinamica delle feature di input in un tensore PyTorch a precisione singola (float32)
    X_tensor = torch.tensor(
        X_test.values if hasattr(X_test, "values") else X_test,
        dtype=torch.float32
    )

    # Disabilita il calcolo dei gradienti per ridurre l'uso della memoria e velocizzare l'inferenza
    with torch.no_grad():
        outputs = model(X_tensor).squeeze()
        y_probs = outputs.numpy()
        # Converte le probabilità in classi binarie (0 o 1) usando la soglia standard di 0.5
        y_pred = (outputs > 0.5).int().numpy()

    # Calcolo analitico di tutte le metriche di classificazione richieste
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_probs)
    }

    # Generazione della matrice di confusione basata sulle classi predette
    cm = confusion_matrix(y_test, y_pred)

    # Output testuale e grafico vincolato al parametro di verbosità
    if verbose:
        print(f"\n{'=' * 60}")
        print(f" NEURAL NETWORK - {title}")
        print(f"{'=' * 60}")

        for k, v in metrics.items():
            print(f"{k:10}: {v:.4f}")

        print("\n Classification Report:\n")
        print(classification_report(y_test, y_pred))

        # Richiamo della funzione interna per la visualizzazione della matrice
        plot_confusion_matrix(cm, title)

    return metrics, cm


def plot_confusion_matrix(cm, title):
    """
    Genera e mostra a schermo il grafico heatmap della matrice di confusione.

    Parametri:
    ----------
    cm : array
        La matrice di confusione da visualizzare.
    title : str
        Il titolo da inserire nell'intestazione del grafico.
    """
    plt.figure(figsize=(6, 4))

    # Rappresentazione grafica tramite seaborn con formattazione a numeri interi (fmt='d')
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


def print_kfold_summary_nn(all_metrics, all_cms):
    """
    Raccoglie i risultati di tutti i fold della Cross-Validation per calcolarne
    la media aritmetica, la deviazione standard e mostrare la matrice media.

    Parametri:
    ----------
    all_metrics : list of dict
        Lista di dizionari contenenti le metriche registrate in ciascun fold.
    all_cms : list of arrays
        Lista delle matrici di confusione ottenute da ogni singolo fold.
    """
    print(f"\n{'=' * 70}")
    print(" NEURAL NETWORK - K-FOLD FINAL SUMMARY")
    print(f"{'=' * 70}")

    avg_metrics = {}

    # Calcolo della media per ciascuna chiave metrica estratta dal primo dizionario disponibile
    for k in all_metrics[0]:
        avg_metrics[k] = np.mean([m[k] for m in all_metrics])

    # Calcolo della deviazione standard sul valore di Accuracy per misurare la stabilità del modello
    std_acc = np.std([m["Accuracy"] for m in all_metrics])

    for k, v in avg_metrics.items():
        print(f"{k:10}: {v:.4f}")

    print(f"\nStd Accuracy: {std_acc:.4f}")

    # Calcolo della matrice di confusione media dividendo la somma per il numero dei fold
    avg_cm = np.mean(all_cms, axis=0)

    print("\n Confusion Matrix Media:\n")
    print(avg_cm)

    # Generazione del grafico della matrice media con visualizzazione decimale (fmt=".2f")
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


def plot_loss_curve(model):
    """
    Traccia l'andamento della loss (funzione di costo) registrata durante
    le epoche di addestramento per verificare la convergenza della rete.

    Parametri:
    ----------
    model : torch.nn.Module
        Il modello contenente l'attributo loss_history popolato nel training.
    """
    plt.figure(figsize=(8, 5))

    # Verifica l'effettiva presenza dello storico prima di procedere al plot grafico
    if hasattr(model, "loss_history"):
        plt.plot(model.loss_history, label="Training Loss")

    plt.title("Loss Curve - Neural Network")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()