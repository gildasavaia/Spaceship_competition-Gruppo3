import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)


def plot_evaluation(y_true, y_pred, y_prob, model_name):
    """Genera i grafici per la Matrice di Confusione e la Curva ROC"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Valutazione Modello: {model_name.upper()}', fontsize=16)

    # 1. Matrice di Confusione
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Falso (Non Trasp.)', 'Vero (Trasp.)'],
                yticklabels=['Falso (Non Trasp.)', 'Vero (Trasp.)'])
    axes[0].set_title('Matrice di Confusione')
    axes[0].set_xlabel('Predizione del Modello')
    axes[0].set_ylabel('Valore Reale')

    # 2. Curva ROC (se le probabilità sono disponibili)
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('Tasso di Falsi Positivi (FPR)')
        axes[1].set_ylabel('Tasso di Veri Positivi (TPR / Recall)')
        axes[1].set_title('Curva ROC')
        axes[1].legend(loc="lower right")
    else:
        axes[1].text(0.5, 0.5, 'Probabilità non disponibili\nper la curva ROC',
                     horizontalalignment='center', verticalalignment='center')
        axes[1].set_title('Curva ROC non generabile')

    plt.tight_layout()
    plt.show()


def main():
    print("=== TRIBUNALE DEI MODELLI - SPACESHIP TITANIC ===\n")

    base_dir = Path(__file__).resolve().parent.parent
    outputs_dir = base_dir / "outputs"
    preprocessed_dir = base_dir / "data" / "preprocessed_folds"

    if not outputs_dir.exists():
        print("Cartella 'outputs' non trovata. Devi prima lanciare i modelli!")
        return

    # Trova tutte le submission create
    submission_files = list(outputs_dir.glob("submission_*.csv"))
    if not submission_files:
        print("Nessun file di predizione trovato in 'outputs'.")
        return

    # Menu dinamico per scegliere quale modello valutare
    print("Modelli disponibili per la valutazione:")
    for i, file_path in enumerate(submission_files, start=1):
        print(f"{i}: {file_path.name}")

    scelta = input(f"\nSeleziona il numero del modello da valutare (1-{len(submission_files)}): ").strip()
    try:
        scelta_idx = int(scelta) - 1
        sub_file = submission_files[scelta_idx]
    except (ValueError, IndexError):
        print("Scelta non valida. Uscita.")
        return

    # Estraiamo le informazioni dal nome del file (es: submission_lgbm_holdout_tree.csv)
    # Rimuoviamo 'submission_' e l'estensione '.csv'
    nome_base = sub_file.stem.replace("submission_", "")

    # Cerca il file delle probabilità corrispondente
    prob_file = outputs_dir / f"prob_{nome_base}.csv"

    # Capiamo quale file di TEST usare per le risposte vere
    # Estraiamo il nome del dataset rimuovendo il prefisso del modello (es: lgbm_)
    parti_nome = nome_base.split('_')
    modello = parti_nome[0]
    dataset_name = "_".join(parti_nome[1:])

    test_file_path = preprocessed_dir / f"{dataset_name}_test.csv"

    if not test_file_path.exists():
        print(f"Errore: Non trovo le risposte corrette! File {test_file_path} mancante.")
        return

    # =========================================================
    # CARICAMENTO DATI
    # =========================================================
    y_pred_df = pd.read_csv(sub_file)
    y_true_df = pd.read_csv(test_file_path)

    y_prob_df = pd.read_csv(prob_file) if prob_file.exists() else None

    # Controllo anti-crash per le submission ufficiali (Kaggle Test Set)
    if 'Transported' not in y_true_df.columns:
        print(f"\n{'-' * 60}")
        print("🏆 DATASET KAGGLE UFFICIALE RILEVATO!")
        print(f"Il file '{sub_file.name}' è basato sul test set finale.")
        print("Poiché non abbiamo le risposte corrette ('Transported'), non possiamo calcolare le metriche qui.")
        print(
            "Questo file è pronto per essere caricato direttamente su Kaggle per ottenere il tuo punteggio in classifica!")
        print(f"{'-' * 60}")
        return

    # =========================================================
    # ALLINEAMENTO DATI (A prova di errore)
    # =========================================================
    # Controlliamo se esiste il PassengerId in entrambi i file
    if 'PassengerId' in y_pred_df.columns and 'PassengerId' in y_true_df.columns:
        y_pred_df = y_pred_df.sort_values('PassengerId').reset_index(drop=True)
        y_true_df = y_true_df.sort_values('PassengerId').reset_index(drop=True)
    else:
        # Se i colleghi hanno rimosso l'ID nel preprocessing, ci fidiamo dell'ordine delle righe
        print("\n[*] Nota: 'PassengerId' assente nel test set. Assumo allineamento sequenziale delle righe.")

    y_true = y_true_df['Transported'].astype(bool)
    y_pred = y_pred_df['Transported'].astype(bool)
    y_prob = y_prob_df['Probability'].values if y_prob_df is not None else None

    # =========================================================
    # CALCOLO METRICHE
    # =========================================================
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{'-' * 50}")
    print(f"REPORT DEL MODELLO: {nome_base.upper()}")
    print(f"{'-' * 50}")
    print(f"🎯 Accuratezza (Kaggle) : {acc:.4f}  (Quanto ci azzecca in totale)")
    print(f"🔎 Precisione           : {prec:.4f}  (Dei previsti Trasportati, quanti lo erano davvero?)")
    print(f"🎣 Sensibilità (Recall) : {rec:.4f}  (Di tutti i veri Trasportati, quanti ne ha trovati?)")
    print(f"⚖️ F1-Score             : {f1:.4f}  (Media armonica tra Precision e Recall)")

    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        print(f"📈 ROC-AUC Score        : {auc:.4f}  (Capacità di separare le classi)")
    print(f"{'-' * 50}")

    # Lanciamo il grafico
    plot_evaluation(y_true, y_pred, y_prob, nome_base)


if __name__ == "__main__":
    main()