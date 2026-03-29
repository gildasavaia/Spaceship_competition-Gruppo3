import pandas as pd
from dataclasses import dataclass

@dataclass
class op5_sumcosts_namesResult:
    df_output: pd.DataFrame


def run_op5(df: pd.DataFrame) -> op5_sumcosts_namesResult:
    """
    OP5 completa:
    - Crea feature TotalSpending
    - Rimuove Names e Surnames
    """

    df_processed = df.copy()

    # Colonne di costo
    cost_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']


    # 1 Feature aggregata
    df_processed['TotalSpending'] = df_processed[cost_cols].sum(axis=1)

    # 2. Rimozione colonne inutili
    colonne_da_rimuovere = ['Names', 'Surnames', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    colonne_esistenti = [col for col in colonne_da_rimuovere if col in df_processed.columns]
    df_processed = df_processed.drop(columns=colonne_esistenti)

    return op5_sumcosts_namesResult(df_output=df_processed)