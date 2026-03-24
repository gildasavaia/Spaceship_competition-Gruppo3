import pandas as pd
from typing import NamedTuple

# Creiamo una NamedTuple per l'output
class SplitDatasetOutputs(NamedTuple):
    df_output: pd.DataFrame

def run_split_dataset(df: pd.DataFrame) -> SplitDatasetOutputs:
    """
    Esegue lo split di colonne specifiche del dataset:
    1. Estrae 'Group' e calcola 'GroupSize' da 'PassengerId'.
    2. Divide 'Cabin' in 'Deck' (Ponte), 'Num' (Numero) e 'Side' (Lato).
    
    Args:
        df (pd.DataFrame): Il DataFrame originale.
        
    Returns:
        SplitDatasetOutputs: Il DataFrame modificato wrappato in una NamedTuple.
    """
    df_mod = df.copy()
    
    # --- 1. Split PassengerId ---
    # Crea la colonna 'Group' (prima metà del PassengerId)
    df_mod['Group'] = df_mod['PassengerId'].str.split('_').str[0]
    # Crea la colonna 'GroupSize' (conteggio delle persone nello stesso gruppo)
    df_mod['GroupSize'] = df_mod.groupby('Group')['Group'].transform('count')
    
    # --- 2. Split Cabin ---
    # Dividiamo la colonna Cabin usando '/' come separatore.
    df_mod[['Deck', 'Num', 'Side']] = df_mod['Cabin'].str.split('/', expand=True)
    
    df_mod[['Names', 'Surnames']] = df_mod['Name'].str.split(' ', expand=True)
    
    print("Feature engineering completato: colonne PassengerId e Cabin divise con successo.")
    
    # spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    # df_mod['TotalSpent'] = df_mod[spending_cols].fillna(0).sum(axis=1)

    # cols_to_drop = spending_cols + ['PassengerId', 'Cabin', 'Name']
    # df_mod.drop(columns=cols_to_drop, inplace=True)
    
    # Metti 'Group' e 'GroupSize' in testa e 'Transported' in fondo
    cols = list(df_mod.columns)
    start = [c for c in ['Group', 'GroupSize'] if c in cols]
    end = ['Transported'] if 'Transported' in cols else []
    middle = [c for c in cols if c not in start + end]
    df_mod = df_mod[start + middle + end]

    return SplitDatasetOutputs(df_output=df_mod)