import unittest
import pandas as pd
import sys
from pathlib import Path

# Aggiungiamo il percorso per importare gli script dei tuoi compagni
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir / "preprocessing"))

from op3_split_dataset import run_split_dataset

""" TEST 1: Feature Engineering & Split Colonne (op3)
Obiettivo: Verificare che le colonne composte come 'Cabin' e 'PassengerId' vengano correttamente
frammentate in sotto-variabili predittive (Deck, Num, Side). Assicura inoltre che la logica di
conteggio dei membri dello stesso gruppo (GroupSize) sia esatta e che le colonne originali vengano
eliminate."""

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Questo mini-dataset finto viene ricreato fresco prima di ogni test
        self.dummy_data = pd.DataFrame({
            'PassengerId': ['0001_01', '0002_01', '0002_02'],
            'Cabin': ['B/1/S', 'F/2/P', 'F/2/P'],
            'Name': ['Mario Rossi', 'Luigi Verdi', 'Peach Rosa'],
            'Transported': [True, False, True]
        })

    def test_split_dataset_logic(self):
        """Verifica che Cabin e PassengerId vengano divisi correttamente."""

        # Eseguiamo il codice dei compagni
        # Passiamo i_train=1 (come se fosse il train)
        risultato = run_split_dataset(self.dummy_data, i_train=1)
        df_out = risultato.df_output

        # 1. Verifica che Cabin sia stata droppata
        self.assertNotIn('Cabin', df_out.columns, "Errore: La colonna Cabin originale non è stata droppata!")

        # 2. Verifica i valori creati dallo split di Cabin
        self.assertEqual(df_out.loc[0, 'Deck'], 'B')
        self.assertEqual(df_out.loc[0, 'Num'], 1.0)
        self.assertEqual(df_out.loc[0, 'Side'], 'S')

        # 3. Verifica la logica del GroupSize (i passeggeri 0002_01 e 0002_02 sono nello stesso gruppo)
        self.assertEqual(df_out.loc[0, 'GroupSize'], 1, "Il passeggero 1 è solo, GroupSize dovrebbe essere 1")
        self.assertEqual(df_out.loc[1, 'GroupSize'], 2,
                         "I passeggeri 2 e 3 viaggiano insieme, GroupSize dovrebbe essere 2")

if __name__ == '__main__':
    unittest.main()