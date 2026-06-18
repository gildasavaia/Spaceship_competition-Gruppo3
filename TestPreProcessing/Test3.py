import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# =====================================================================
# 1. SETUP DEI PERCORSI
# =====================================================================
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir / "preprocessing"))

from op4_handler_nullvalue import run_handle_null_values

""" TEST 3: Imputazione Logica e Statistica dei Valori Nulli (op4)
Obiettivo: Assicurarsi che nessun valore NaN sopravviva al preprocessing. Il test valida
l'applicazione rigorosa delle regole matematiche (media per l'età, 0.0 per i costi mancanti) e le
regole logiche di dominio (es. se un passeggero si trova nel ponte A, il suo pianeta d'origine deve
essere forzato a 'Europa')."""

class TestNullImputation(unittest.TestCase):

    def test_imputation_logic(self):
        """Verifica la corretta imputazione dei NaN e le regole matematiche/di dominio."""

        dummy_data = pd.DataFrame({
            'PassengerId': ['0001_01', '0002_01', '0003_01', '0004_01'],
            'HomePlanet': ['Earth', 'Earth', 'Mars', np.nan],
            'Age': [20.0, 30.0, 40.0, np.nan],
            # Usiamo le stringhe per evitare il FutureWarning di Pandas
            'CryoSleep': ['False', 'False', 'True', 'False'],
            'RoomService': [100.0, 200.0, np.nan, np.nan],
            'Destination': ['TRAPPIST-1e', 'TRAPPIST-1e', 'PSO', 'TRAPPIST-1e'],
            'VIP': ['False', 'False', 'False', 'False'],
            # ECCO LA CORREZIONE: Mettiamo il passeggero 4 nel Deck 'A'!
            'Deck': ['F', 'F', 'F', 'A'],
            'Num': ['1', '2', '3', '4'],
            'Side': ['P', 'P', 'S', 'P'],
            'Surnames': ['Rossi', 'Bianchi', 'Verdi', 'Neri']
        })

        risultato = run_handle_null_values(dummy_data.copy())
        df_clean = risultato.df_output

        # A) CONTROLLO GLOBALE
        somma_nan = df_clean.isnull().sum().sum()
        self.assertEqual(somma_nan, 0, f"🚨 Errore: Ci sono ancora {somma_nan} valori nulli nel DataFrame!")

        # B) CONTROLLO NUMERICO (Media)
        self.assertEqual(df_clean.loc[3, 'Age'], 30.0,
                         "Errore: L'età NaN non è stata riempita con la media corretta (30.0).")

        # C) CONTROLLO COSTI
        self.assertEqual(df_clean.loc[2, 'RoomService'], 0.0,
                         "Errore: Spesa NaN in Cryosleep non azzerata.")
        self.assertEqual(df_clean.loc[3, 'RoomService'], 0.0,
                         "Errore: Spesa NaN normale non azzerata.")

        # D) CONTROLLO REGOLA LOGICA HOMEPLANET
        # Avendo Deck 'A', il codice DEVE ignorare la probabilità e forzare 'Europa'
        self.assertEqual(df_clean.loc[3, 'HomePlanet'], 'Europa',
                         "Errore: Il passeggero nel ponte 'A' non è stato assegnato ad 'Europa'!")


if __name__ == '__main__':
    unittest.main()