import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)


class MetricsEvaluator:
    """
    Classe OOP dedicata al calcolo e alla visualizzazione grafica delle metriche.
    """

    def __init__(self, y_true, y_pred, y_probs=None, dataset_name="Dataset"):
        self.y_true = y_true.astype(bool)
        self.y_pred = y_pred.astype(bool)
        self.y_probs = y_probs
        self.dataset_name = dataset_name

    def calculate_metrics(self):
        """Calcola le metriche testuali."""
        metrics = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision': precision_score(self.y_true, self.y_pred),
            'recall': recall_score(self.y_true, self.y_pred),
            'f1': f1_score(self.y_true, self.y_pred)
        }
        if self.y_probs is not None:
            metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_probs)
        else:
            metrics['roc_auc'] = None
        return metrics

    def print_report(self):
        """Stampa il report testuale in console."""
        m = self.calculate_metrics()
        print(f"\n{'-' * 50}")
        print(f" 📊 REPORT METRICHE: {self.dataset_name.upper()}")
        print(f"{'-' * 50}")
        print(f"   🎯 Accuracy  : {m['accuracy']:.4f}")
        print(f"   🔎 Precision : {m['precision']:.4f}")
        print(f"   🎣 Recall    : {m['recall']:.4f}")
        print(f"   ⚖️ F1-Score  : {m['f1']:.4f}")
        if m['roc_auc'] is not None:
            print(f"   📈 ROC-AUC   : {m['roc_auc']:.4f}")
        print(f"{'-' * 50}\n")

    def plot_visuals(self):
        """Genera i grafici: Matrice di Confusione e Curva ROC."""
        print(f"[*] Generazione grafici per {self.dataset_name}... Chiudi la finestra per continuare.")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"Valutazione Modello: {self.dataset_name}", fontsize=16)

        # 1. Matrice di Confusione
        cm = confusion_matrix(self.y_true, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Matrice di Confusione')
        axes[0].set_xlabel('Predetto')
        axes[0].set_ylabel('Reale')

        # 2. Curva ROC
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

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()