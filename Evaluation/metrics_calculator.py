from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class MetricsEvaluator:
    """
    Classe OOP dedicata al calcolo e alla stampa delle metriche dei modelli.
    Può essere istanziata da qualsiasi modello (LightGBM, SVC, RF, ecc.)
    """

    def __init__(self, y_true, y_pred, y_probs=None, dataset_name="Dataset"):
        # Inizializziamo l'oggetto convertendo tutto in booleano per sicurezza
        self.y_true = y_true.astype(bool)
        self.y_pred = y_pred.astype(bool)
        self.y_probs = y_probs
        self.dataset_name = dataset_name

    def calculate_metrics(self):
        """Calcola le metriche e le restituisce come dizionario."""
        metrics = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision': precision_score(self.y_true, self.y_pred),
            'recall': recall_score(self.y_true, self.y_pred),
            'f1': f1_score(self.y_true, self.y_pred)
        }

        # Il ROC-AUC richiede le probabilità. Se ci sono, lo calcola.
        if self.y_probs is not None:
            metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_probs)
        else:
            metrics['roc_auc'] = None

        return metrics

    def print_report(self):
        """Metodo per stampare il report formattato a schermo."""
        m = self.calculate_metrics()

        print(f"\n{'-' * 50}")
        print(f" 📊 METRICHE IN TEMPO REALE: {self.dataset_name.upper()}")
        print(f"{'-' * 50}")
        print(f"   🎯 Accuracy  : {m['accuracy']:.4f}")
        print(f"   🔎 Precision : {m['precision']:.4f}")
        print(f"   🎣 Recall    : {m['recall']:.4f}")
        print(f"   ⚖️ F1-Score  : {m['f1']:.4f}")

        if m['roc_auc'] is not None:
            print(f"   📈 ROC-AUC   : {m['roc_auc']:.4f}")
        print(f"{'-' * 50}\n")