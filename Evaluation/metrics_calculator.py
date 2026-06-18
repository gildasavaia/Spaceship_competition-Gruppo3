import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

""" Questo script definisce la classe 'MetricsEvaluator'. Qualsiasi modello venga addestrato passa le sue risposte a
questa classe, garantendo che tutti vengano giudicati esattamente con lo stesso metro di misura.

1) Evita la duplicazione del codice di valutazione nei vari script.
2) Contiene la logica per impacchettare i risultati in un file JSON temporaneo, permettendo all'Orchestrator di leggerli 
   e compilare la Leaderboard di sessione.
3) Automatizza la creazione di Matrici di Confusione e Curve ROC."""


class MetricsEvaluator:
    """ Classe dedicata al calcolo, alla visualizzazione grafica e all'esportazione delle metriche per l'Orchestrator."""


    def __init__(self, y_true, y_pred, y_probs=None, dataset_name="Dataset"):
        """ Questa funzione inizializza l'oggetto salvando le risposte vere e quelle predette. In particolare
        .astype(bool) garantisce che i dati siano nel formato corretto (True/False o 1/0) per evitare errori matematici7
        durante il calcolo."""

        self.y_true = y_true.astype(bool)
        self.y_pred = y_pred.astype(bool)
        self.y_probs = y_probs
        self.dataset_name = dataset_name


    def calculate_metrics(self):
        """Calcola le metriche matematiche e le restituisce in un dizionario."""

        metrics = {
            'Model': self.dataset_name,  # Aggiungiamo il nome del modello!
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision': precision_score(self.y_true, self.y_pred),
            'recall': recall_score(self.y_true, self.y_pred),
            'f1': f1_score(self.y_true, self.y_pred)
        }

        # L'AUC calcola la capacità del modello di separare le due classi. Richiede le probabilità continue. Se non ci
        # sono, assegniamo None per non far crashare lo script.
        if self.y_probs is not None:
            metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_probs)
        else:
            metrics['roc_auc'] = None
        return metrics


    def export_to_orchestrator(self):
        """ Salva un file temporaneo affinché l'Orchestrator possa raccoglierlo. Dato che i modelli girano in processi
        isolati, questo file JSON è l'unico modo per far comunicare il modello finito con il menu principale."""

        m = self.calculate_metrics()

        # Troviamo la cartella 'outputs' generale in modo dinamico.
        base_dir = Path(__file__).resolve().parent.parent
        out_dir = base_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Definiamo il percorso del file di scambio.
        temp_file = out_dir / "temp_metrics.json"

        # Salviamo fisicamente i dati su disco. L'orchestrator li leggerà creando la leaderboard e poi lo distruggerà.
        with open(temp_file, "w") as f:
            json.dump(m, f)


    def print_report(self):
        """Stampa il report testuale formattato in console ed innesca l'esportazione dei dati."""

        m = self.calculate_metrics()

        # Passiamo le metriche all'orchestratore in modo automatico.
        self.export_to_orchestrator()

        print(f"\n{'-' * 50}")
        print(f" REPORT METRICHE: {self.dataset_name.upper()}")
        print(f"{'-' * 50}")
        print(f"   Accuracy  : {m['accuracy']:.4f}")
        print(f"   Precision : {m['precision']:.4f}")
        print(f"   Recall    : {m['recall']:.4f}")
        print(f"   F1-Score  : {m['f1']:.4f}")
        if m['roc_auc'] is not None:
            print(f"   ROC-AUC   : {m['roc_auc']:.4f}")
        print(f"{'-' * 50}\n")


    def plot_visuals(self):
        """Genera i grafici visivi (Matrice di Confusione e Curva ROC) per l'analisi umana."""

        print(f"[*] Generazione grafici per {self.dataset_name}... Chiudi la finestra per continuare.")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"Valutazione Modello: {self.dataset_name}", fontsize=16)

        # 1. Matrice di Confusione, ,ostra i Veri Positivi, Falsi Positivi, Veri Negativi e Falsi Negativi.
        cm = confusion_matrix(self.y_true, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Matrice di Confusione')
        axes[0].set_xlabel('Predetto')
        axes[0].set_ylabel('Reale')

        # 2. Curva ROC, mostra il compromesso tra la capacità di trovare i positivi veri (TPR) e gli allarmi falsi (FPR).
        if self.y_probs is not None:
            fpr, tpr, _ = roc_curve(self.y_true, self.y_probs)
            auc_val = roc_auc_score(self.y_true, self.y_probs)
            axes[1].plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_val:.4f})", color='darkorange', lw=2)
            axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[1].set_title('Curva ROC')
            axes[1].set_xlabel('False Positive Rate')
            axes[1].set_ylabel('True Positive Rate')
            axes[1].legend(loc="lower right")
        else:
            axes[1].text(0.5, 0.5, "Probabilità non disponibili\nper la curva ROC",
                         ha='center', va='center', fontsize=12)

        # Ottimizza gli spazi vuoti tra i grafici e mostra l'immagine a schermo
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()