import unittest
import pandas as pd
import sys
from pathlib import Path

# =====================================================================
# 1. SETUP DEI PERCORSI
# =====================================================================
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir / "preprocessing"))

# =====================================================================
# 2. IMPORTAZIONE DEL MODULO
# =====================================================================
# Nota: Assicurati che 'run_sumcosts_names' sia il nome corretto della funzione in op5
from op5_sumcosts_names import run_op5

""" TEST 5: Ingegneria Finanziaria e Barriera di Sicurezza (op5)
Obiettivo: Verificare l'esattezza matematica della nuova feature 'TotalSpending' (somma di
RoomService, FoodCourt, ShoppingMall, Spa, VRDeck). Funge inoltre da barriera per assicurare che la
colonna testuale 'Names' venga rigorosamente distrutta prima di raggiungere i modelli di Machine
Learning."""

class TestForbiddenColumnsAndCosts(unittest.TestCase):

    def test_sumcosts_and_drop_names(self):
        """Verifica la creazione corretta di TotalSpending e l'eliminazione della colonna testuale Names."""

        # 1. Creiamo un dataset finto con la "colonna vietata" e tutti i costi
        dummy_data = pd.DataFrame({
            'PassengerId': ['0001_01', '0002_01'],
            'Names': ['Mario', 'Luigi'],  # <--- QUESTA DEVE SPARIRE!
            'Surnames': ['Rossi', 'Verdi'],  # <--- Questa deve restare per ora (si elimina in op9)
            'RoomService': [100.0, 0.0],
            'FoodCourt': [50.0, 20.0],
            'ShoppingMall': [0.0, 0.0],
            'Spa': [200.0, 10.0],
            'VRDeck': [0.0, 50.0]
        })

        # 2. Eseguiamo la funzione
        # (Se richiede parametri aggiuntivi, aggiungili qui, ma in base al README dovrebbe bastare il df)
        risultato = run_op5(dummy_data.copy())

        # 3. Recuperiamo il dataframe in uscita
        df_out = risultato.df_output

        # =====================================================================
        # 4. LE ASSERZIONI (PULIZIA E MATEMATICA)
        # =====================================================================

        # A) TRAPPOLA SULLE COLONNE VIETATE
        self.assertNotIn('Names', df_out.columns,
                         "🚨 Errore Fatale: La colonna 'Names' non è stata eliminata! Farà crashare i modelli.")

        # B) CONTROLLO DI SICUREZZA
        # Surnames deve sopravvivere a questo step (viene usata in OP4/OP9)
        self.assertIn('Surnames', df_out.columns,
                      "Errore: La colonna 'Surnames' è stata eliminata per sbaglio in questo modulo!")

        # C) CONTROLLO MATEMATICO SULLA NUOVA FEATURE
        self.assertIn('TotalSpending', df_out.columns,
                      "Errore: La feature 'TotalSpending' non è stata creata.")

        # Il passeggero 1 ha speso: 100 + 50 + 0 + 200 + 0 = 350.0
        self.assertEqual(df_out.loc[0, 'TotalSpending'], 350.0,
                         "Errore: La somma dei costi del Passeggero 1 è sbagliata.")

        # Il passeggero 2 ha speso: 0 + 20 + 0 + 10 + 50 = 80.0
        self.assertEqual(df_out.loc[1, 'TotalSpending'], 80.0,
                         "Errore: La somma dei costi del Passeggero 2 è sbagliata.")


if __name__ == '__main__':
    unittest.main()