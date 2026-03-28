import pandas as pd
from dataclasses import dataclass

@dataclass
class op5_sumcosts_namesResult:
    df_output: pd.DataFrame


def run_op5(df: pd.DataFrame) -> op5_sumcosts_namesResult:
    """
    OP5 completa:
    - Sostituisce NaN nei costi con 0
    - Crea feature TotalSpending
    - Rimuove Names e Surnames
    """

    df_processed = df.copy()

    # Colonne di costo
    cost_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    # 1. Pulizia NaN
    df_processed[cost_cols] = df_processed[cost_cols].fillna(0)

    # 2. Feature aggregata
    df_processed['TotalSpending'] = df_processed[cost_cols].sum(axis=1)

    # 3. Rimozione colonne inutili
    colonne_da_rimuovere = ['Names', 'Surnames']
    colonne_esistenti = [col for col in colonne_da_rimuovere if col in df_processed.columns]
    df_processed = df_processed.drop(columns=colonne_esistenti)

    return op5_sumcosts_namesResult(df_output=df_processed)