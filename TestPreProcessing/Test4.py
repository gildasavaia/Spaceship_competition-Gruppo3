import unittest
import pandas as pd
import sys
from pathlib import Path

# =====================================================================
# 1. SETUP DEI PERCORSI
# =====================================================================
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir / "preprocessing"))

# Importiamo la funzione dal file op8_encoding
from op8_encoding import run_encoding

""" TEST 4: Robustezza One-Hot Encoding e Allineamento (op9)
Obiettivo: Simulazione di uno scenario di stress in cui il Test set contiene categorie aliene mai
viste nel Train set (es. pianeta 'Plutone'). Garantisce che l'encoder non vada in crash, prevenga
la Dummy Variable Trap e restituisca Train e Test set con l'identico numero di colonne."""

class TestEncodingRobustness(unittest.TestCase):

    def test_encoding_and_column_drops(self):
        """Testa la resistenza dell'OHE a categorie sconosciute e l'eliminazione delle colonne ridondanti."""

        # 1. Dataset di TRAIN finto
        # Nota: Qui mettiamo solo Earth e Mars, e solo Deck A e B
        train_df = pd.DataFrame({
            'Group': ['0001', '0002'],
            'Surnames': ['Rossi', 'Bianchi'],
            'HomePlanet': ['Earth', 'Mars'],
            'CryoSleep': ['True', 'False'],
            'Destination': ['TRAPPIST-1e', 'PSO J318.5-22'],
            'VIP': ['False', 'True'],
            'Deck': ['A', 'B'],
            'Side': ['P', 'S'],
            'Age': [25.0, 30.0] # Mettiamo un valore numerico per vedere se viene toccato
        })

        # 2. Dataset di TEST finto con una TRAPPOLA ALIENA
        test_df = pd.DataFrame({
            'Group': ['0003'],
            'Surnames': ['Verdi'],
            # LA TRAPPOLA: 'Plutone' e 'Deck C' NON esistono nel dataset di Train!
            'HomePlanet': ['Plutone'],
            'CryoSleep': ['False'],
            'Destination': ['TRAPPIST-1e'],
            'VIP': ['False'],
            'Deck': ['C'],
            'Side': ['P'],
            'Age': [40.0]
        })

        # =====================================================================
        # ESECUZIONE (La prova del nove sul Crash)
        # =====================================================================
        try:
            # Passiamo lo stesso df sia per tree che per nn (per semplificare il test)
            risultato = run_encoding(train_df.copy(), train_df.copy(), test_df.copy(), test_df.copy())
        except Exception as e:
            self.fail(f"🚨 DISASTRO! L'encoder è crashato a causa di una categoria sconosciuta. Errore: {e}")

        # Estraiamo i dataframe della rete neurale (che subiscono l'One-Hot Encoding)
        nn_train = risultato.df_nn_train
        nn_test = risultato.df_nn_test

        # =====================================================================
        # LE ASSERZIONI
        # =====================================================================

        # A) Pulizia Iniziale: Eliminazione colonne vietate
        self.assertNotIn('Group', nn_train.columns, "Errore: 'Group' non è stato rimosso!")
        self.assertNotIn('Surnames', nn_train.columns, "Errore: 'Surnames' non è stato rimosso!")

        # B) Dummy Variable Trap: Eliminazione ridondanze
        self.assertNotIn('CryoSleep_False', nn_train.columns, "Errore: 'CryoSleep_False' non è stato eliminato!")
        self.assertNotIn('VIP_False', nn_train.columns, "Errore: 'VIP_False' non è stato eliminato!")
        self.assertNotIn('Side_S', nn_train.columns, "Errore: 'Side_S' non è stato eliminato!")

        # C) ALLINEAMENTO COLONNE (Il test più importante!)
        # Anche se il Test set aveva il pianeta 'Plutone', l'One-Hot Encoder deve ignorarlo
        # (usando handle_unknown='ignore') e produrre un dataframe del Test con le STESSE
        # IDENTICHE colonne del Train, altrimenti i modelli andranno in crash durante la predizione.
        self.assertListEqual(
            list(nn_train.columns),
            list(nn_test.columns),
            "🚨 ERRORE FATALE: Le colonne del Train e del Test OHE non coincidono! I modelli andranno in crash."
        )

if __name__ == '__main__':
    unittest.main()