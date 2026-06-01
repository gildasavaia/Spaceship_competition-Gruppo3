import pandas as pd
from dataclasses import dataclass

@dataclass
class op5_sumcosts_namesResult:
    df_output: pd.DataFrame


def run_op5(df: pd.DataFrame) -> op5_sumcosts_namesResult:
    """
    OP5 completa:
    - Crea feature TotalSpending con i valori originari
    - Trasforma le colonne di spesa in valori binari (1 se ha speso, 0 se non ha speso)
    - Rimuove solo Names e Surnames
    """

    df_processed = df.copy()

    # Colonne di costo
    cost_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    # 1. Feature aggregata (calcolata prima della binarizzazione)
    # Controlliamo che le colonne esistano per evitare errori
    colonne_costo_esistenti = [col for col in cost_cols if col in df_processed.columns]
    if colonne_costo_esistenti:
        df_processed['TotalSpending'] = df_processed[colonne_costo_esistenti].sum(axis=1)
        
        # Nuove feature di spesa
        df_processed['SpendingZero'] = (df_processed['TotalSpending'] == 0).astype(int)
        
        # Controlliamo la presenza delle colonne specifiche di luxury
        luxury_cols = [c for c in ['Spa', 'VRDeck'] if c in df_processed.columns]
        if luxury_cols:
            df_processed['HasLuxurySpending'] = (df_processed[luxury_cols].sum(axis=1) > 0).astype(int)

    # Nuova feature: IsAlone (1 se GroupSize == 1)
    if 'GroupSize' in df_processed.columns:
        df_processed['IsAlone'] = (df_processed['GroupSize'] == 1).astype(int)

    # # 2. Trasformazione delle colonne di costo in valori binari (0 e 1)
    # for col in colonne_costo_esistenti:
    #     # Se il valore è maggiore di 0 restituisce True (1), altrimenti False (0)
    #     df_processed[col] = (df_processed[col] > 0).astype(int)

    # 3. Rimozione colonne inutili (solo i nomi)
    colonna_da_rimuovere = ['Names']
    colonne_esistenti_da_rimuovere = [col for col in colonna_da_rimuovere if col in df_processed.columns]
    df_processed = df_processed.drop(columns=colonne_esistenti_da_rimuovere)
    # colonna_da_rimuovere = ['Surnames']
    # colonne_esistenti_da_rimuovere = [col for col in colonna_da_rimuovere if col in df_processed.columns]
    # df_processed = df_processed.drop(columns=colonne_esistenti_da_rimuovere)

    return op5_sumcosts_namesResult(df_output=df_processed)