import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)


def run_full_evaluation(model, X_test, y_test, title="Valutazione Modello", verbose=True, is_nn=False):
    """
    Esegue la valutazione completa delle performance del modello (Alberi o Rete Neurale).
    Calcola le metriche fondamentali di classificazione binaria, mostra la
    matrice di confusione ed eventualmente traccia la curva ROC (solo per modelli Tree).

    Parametri:
    ----------
    model : oggetto modello / torch.nn.Module
        Il classificatore (XGBoost, CatBoost o PyTorch NN) addestrato da valutare.
    X_test : DataFrame o array-like
        La matrice delle feature del set di test.
    y_test : Series o array-like
        I target reali relativi al set di test.
    title : str, opzionale
        Il titolo identificativo da utilizzare nei report e nei grafici.
    verbose : bool, opzionale
        Se impostato su True, stampa i risultati a schermo e genera i grafici.
    is_nn : bool, opzionale
        Flag da impostare a True se il modello analizzato è una Rete Neurale PyTorch.

    Restituisce:
    -----------
    metrics : dict
        Dizionario contenente i valori di accuracy, precision, recall, f1 ed eventualmente roc_auc.
    cm : array
        La matrice di confusione calcolata.
    """
    # 1. GENERAZIONE DELLE PREDIZIONI PUNTUALI E DELLE PROBABILITÀ A SECONDA DEL MODELLO
    if is_nn:
        # Imposta la rete in modalità valutazione per disattivare il Dropout e la Batch Normalization
        model.eval()

        # Conversione dinamica delle feature di input in un tensore PyTorch a precisione singoma (float32)
        X_tensor = torch.tensor(
            X_test.values if hasattr(X_test, "values") else X_test,
            dtype=torch.float32
        )

        # Disabilita il calcolo dei gradienti per ridurre l'uso della memoria e velocizzare l'inferenza
        with torch.no_grad():
            outputs = model(X_tensor).squeeze()
            y_probs = outputs.cpu().numpy()
            # Converte le probabilità in classi binarie (0 o 1) usando la soglia standard di 0.5
            y_pred = (outputs > 0.5).int().cpu().numpy()

        # Forza y_test a tipo nativo intero/bool coerente per il confronto con la NN
        y_test_clean = np.array(y_test).astype(int)
    else:
        # Generazione delle predizioni puntuali e delle probabilità stimate per la classe positiva
        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # Standardizzazione del target e delle predizioni in formato booleano per evitare anomalie di formato
        y_test_clean = np.array(y_test).astype(bool)
        if isinstance(y_pred[0], str):
            y_pred = np.array([True if str(x).lower() == 'true' else False for x in y_pred])
        else:
            y_pred = y_pred.astype(bool)

    # 2. CALCOLO ANALITICO DI TUTTE LE METRICHE DI CLASSIFICAZIONE RICHIESTE
    acc = accuracy_score(y_test_clean, y_pred)
    prec = precision_score(y_test_clean, y_pred, zero_division=0)
    rec = recall_score(y_test_clean, y_pred, zero_division=0)
    f1 = f1_score(y_test_clean, y_pred, zero_division=0)
    cm = confusion_matrix(y_test_clean, y_pred)

    # Popolamento iniziale del dizionario con mappatura doppia (minuscola/maiuscola) per retrocompatibilità
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    }

    # Calcolo della metrica ROC-AUC vincolato solo ai modelli ad albero (XGBoost / CatBoost)
    if not is_nn:
        roc_auc = roc_auc_score(y_test_clean, y_probs)
        metrics['roc_auc'] = roc_auc
        metrics['AUC-ROC'] = roc_auc

    # 3. STAMPA TESTUALE E VISUALIZZAZIONE DEI GRAFICI VINCOLATA AL FLAG VERBOSE
    if verbose:
        print(f"\n{'-' * 60}")
        print(f" REPORT METRICHE: {title.upper()}")
        print(f"{'-' * 60}")
        print(f"   Accuracy  : {acc:.4f}")
        print(f"   Precision : {prec:.4f}")
        print(f"   Recall    : {rec:.4f}")
        print(f"   F1-Score  : {f1:.4f}")
        if not is_nn:
            print(f"   ROC-AUC   : {metrics['roc_auc']:.4f}")
        print(f"{'-' * 60}")

        if is_nn:
            print("\n Classification Report:\n")
            print(classification_report(y_test_clean, y_pred))

        # RENDERING GRAFICO DIFFERENZIATO IN BASE AL MODELLO
        if is_nn:
            # Per la NN genera un singolo grafico dedicato unicamente alla Matrice di Confusione
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=["False", "True"], yticklabels=["False", "True"])
            plt.title(f'Confusion Matrix - {title}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.show()

            # Richiamo automatico della curva di Loss registrata durante il training della NN
            plot_loss_curve(model)
        else:
            # Configurazione dell'area di disegno per ospitare i due grafici affiancati (Alberi)
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f"{title}", fontsize=16)

            # Matrice di Confusione (formato intero con heatmap blu)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                        xticklabels=["False", "True"], yticklabels=["False", "True"])
            axes[0].set_title('Matrice di Confusione')
            axes[0].set_xlabel('Predetto')
            axes[0].set_ylabel('Reale')

            # Calcolo dei punti della curva ROC e plot grafico rispetto alla bisettrice casuale
            fpr, tpr, _ = roc_curve(y_test_clean, y_probs)
            axes[1].plot(fpr, tpr, label=f"ROC Curve (AUC = {metrics['roc_auc']:.4f})", color='darkorange', lw=2)
            axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[1].set_title('Curva ROC')
            axes[1].set_xlabel('False Positive Rate')
            axes[1].set_ylabel('True Positive Rate')
            axes[1].legend(loc="lower right")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    return metrics, cm


def print_kfold_summary(fold_metrics, fold_confusion_matrices, is_nn=False):
    """
    Raccoglie i risultati di tutti i fold della Cross-Validation per calcolarne
    la media aritmetica delle performance e mostra la matrice di confusione aggregata.

    Parametri:
    ----------
    fold_metrics : list of dict
        Lista dei dizionari contenenti le metriche registrate per ogni fold.
    fold_confusion_matrices : list of arrays
        Lista delle matrici di confusione generate in ciascun fold.
    is_nn : bool, opzionale
        Flag da impostare a True se l'analisi fa riferimento alla Rete Neurale.
    """
    print("\n" + "=" * 60)
    print(f" SINTESI RISULTATI K-FOLD ({'NEURAL NETWORK' if is_nn else 'TREE MODEL'}) (Medie)")
    print("=" * 60)

    # Estrazione flessibile delle chiavi basata sulle convenzioni dei vecchi file main
    acc_key = 'Accuracy' if (is_nn or 'Accuracy' in fold_metrics[0]) else 'accuracy'
    prec_key = 'Precision' if (is_nn or 'Precision' in fold_metrics[0]) else 'precision'
    rec_key = 'Recall' if (is_nn or 'Recall' in fold_metrics[0]) else 'recall'
    f1_key = 'F1-Score' if (is_nn or 'F1-Score' in fold_metrics[0]) else 'f1'

    # Calcolo della media aritmetica per ciascuna metrica monitorata
    avg_acc = np.mean([m[acc_key] for m in fold_metrics])
    avg_prec = np.mean([m[prec_key] for m in fold_metrics])
    avg_rec = np.mean([m[rec_key] for m in fold_metrics])
    avg_f1 = np.mean([m[f1_key] for m in fold_metrics])

    print(f"   Accuracy Media  : {avg_acc:.4f}")
    print(f"   Precision Media : {avg_prec:.4f}")
    print(f"   Recall Media    : {avg_rec:.4f}")
    print(f"   F1-Score Medio  : {avg_f1:.4f}")

    # Visualizzazione della metrica ROC-AUC esclusiva per gli alberi di decisione
    if not is_nn:
        roc_key = 'AUC-ROC' if 'AUC-ROC' in fold_metrics[0] else 'roc_auc'
        if roc_key in fold_metrics[0]:
            avg_roc = np.mean([m[roc_key] for m in fold_metrics])
            print(f"   ROC-AUC Medio   : {avg_roc:.4f}")

    # Calcolo della deviazione standard sull'accuratezza per misurare la stabilità (solo per NN)
    if is_nn:
        std_acc = np.std([m[acc_key] for m in fold_metrics])
        print(f"   Std Accuracy    : {std_acc:.4f}")

    print("=" * 60 + "\n")

    # Rendering e aggregazione della matrice finale secondo i vecchi standard strutturali
    if is_nn:
        # Calcolo della matrice di confusione media dividendo la somma per il numero dei fold
        agg_cm = np.mean(fold_confusion_matrices, axis=0)
        fmt_type = ".2f"
        title_text = "Confusion Matrix Media - NN"
    else:
        # Somma algebrica delle singole matrici di ogni fold per ottenere la ripartizione totale
        agg_cm = np.sum(fold_confusion_matrices, axis=0)
        fmt_type = "d"
        title_text = "Matrice di Confusione Cumulativa (Tutti i Fold)"

    # Plot grafico della heatmap finale aggregata
    plt.figure(figsize=(7, 5))
    sns.heatmap(agg_cm, annot=True, fmt=fmt_type, cmap='Blues',
                xticklabels=["False", "True"], yticklabels=["False", "True"])
    plt.title(title_text)
    plt.xlabel("Predicted" if is_nn else "Predetto")
    plt.ylabel("Actual" if is_nn else "Reale")
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
    # Verifica l'effettiva presenza dello storico prima di procedere al plot grafico
    if hasattr(model, "loss_history") and model.loss_history:
        plt.figure(figsize=(8, 5))
        plt.plot(model.loss_history, label="Training Loss", color='crimson', lw=2)
        plt.title("Loss Curve - Neural Network")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Storico della loss non trovato o vuoto sul modello NN.")