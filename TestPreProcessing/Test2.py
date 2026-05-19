import unittest
import pandas as pd
import sys
from pathlib import Path

# =====================================================================
# 1. DIAMO A PYTHON LE INDICAZIONI STRADALI
# =====================================================================
# Troviamo la cartella principale del progetto e aggiungiamo 'preprocessing' al percorso
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir / "preprocessing"))

# =====================================================================
# 2. ORA POSSIAMO IMPORTARE (perché Python sa dove cercare!)
# =====================================================================
from op8_scaling import run_scaling

""" TEST 2: Prevenzione Data Leakage nello Scaling (op8)
Obiettivo: Garantire che lo StandardScaler non "sbirci" i dati del futuro. Verifica matematicamente
che il calcolo della media e varianza (fit) avvenga esclusivamente sul Train set. Il Test set deve
essere scalato passivamente, prevenendo un inquinamento che falserebbe completamente le metriche
dei modelli."""

class TestDataLeakage(unittest.TestCase):

    def test_no_data_leakage_in_scaling(self):
        """Verifica che lo scaler venga 'addestrato' (fit) SOLO sul Train set."""

        # 1. Creiamo un Train set finto con valori numerici molto piccoli
        train_df = pd.DataFrame({
            'PassengerId': ['1', '2', '3'],
            'Age': [20.0, 25.0, 30.0],  # Media = 25
            'RoomService': [0.0, 10.0, 20.0]
        })

        # 2. Creiamo un Test set finto con valori numerici GIGANTESCHI
        test_df = pd.DataFrame({
            'PassengerId': ['4', '5', '6'],
            'Age': [80.0, 90.0, 100.0],  # Media = 90
            'RoomService': [1000.0, 2000.0, 3000.0]
        })

        # Eseguiamo la funzione dei compagni passando copie per sicurezza
        risultato = run_scaling(train_df.copy(), test_df.copy())

        # Recuperiamo i dataset standardizzati (dal vostro README so che si chiamano così)
        train_scaled = risultato.df_nn_train
        test_scaled = risultato.df_nn_test

        # IL CONTROLLO MATEMATICO SUL TRAIN:
        # Lo StandardScaler porta sempre la media a 0 sui dati su cui fa il ".fit()".
        media_train_age = train_scaled['Age'].mean()
        self.assertAlmostEqual(media_train_age, 0.0, places=5,
                               msg="Errore: Il Train set non è stato scalato correttamente a media 0.")

        # LA TRAPPOLA SUL TEST (DATA LEAKAGE):
        # Se hanno fittato anche sul test, la media sarà 0. Se hanno fatto bene, sarà un numero alto!
        media_test_age = test_scaled['Age'].mean()

        self.assertNotAlmostEqual(media_test_age, 0.0, places=1,
                                  msg="🚨 ALLARME DATA LEAKAGE! Lo scaler ha calcolato la media sul Test set invece di usare quella del Train!")


if __name__ == '__main__':
    unittest.main()