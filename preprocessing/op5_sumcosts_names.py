import pandas as pd
from dataclasses import dataclass

@dataclass
class op5_sumcosts_namesResult:
    df_output: pd.DataFrame

def run_remove_names(df: pd.DataFrame) -> op5_sumcosts_namesResult:
    """
    Rimuove le features 'Names' e 'Surnames' dal dataset.
    """
    # Creiamo una copia per non modificare il dataframe originale inplace
    df_processed = df.copy()
    
    # Lista delle colonne da eliminare
    colonne_da_rimuovere = ['Names', 'Surnames']
    
    # Elimina le colonne (controllando prima che esistano per evitare errori)
    colonne_esistenti = [col for col in colonne_da_rimuovere if col in df_processed.columns]
    df_processed = df_processed.drop(columns=colonne_esistenti)
    
    return op5_sumcosts_namesResult(df_output=df_processed)


#Gestione dei valori NaN all'interno dei costi 
def pulisci_costi_nulli(df):
    """
    Operazione 5: Sostituisce i valori nulli (NaN) delle features relative
    ai costi extra con 0.
    """
    cost_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # Sostituisce i NaN con 0
    df[cost_cols] = df[cost_cols].fillna(0)
    
    # Se ti serve anche il TotalSpending, puoi aggiungerlo qui:
    # df['TotalSpending'] = df[cost_cols].sum(axis=1)
    
    return df