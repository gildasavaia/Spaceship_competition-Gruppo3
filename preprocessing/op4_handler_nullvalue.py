import pandas as pd
import numpy as np
from typing import NamedTuple

# Creiamo una NamedTuple per l'output, seguendo lo standard della tua pipeline
class HandleNullValueOutputs(NamedTuple):
    df_output: pd.DataFrame

def impute_group_mode(df: pd.DataFrame, columns_to_impute: list, group_col: str = 'Group') -> pd.DataFrame:
    """
    Fase 1: Sostituisce i valori nulli con la moda all'interno dello stesso gruppo.
    """
    print(f"\n--- Avvio Imputazione basata sul Gruppo ({group_col}) ---")
    df_filled = df.copy()
    
    for col in columns_to_impute:
        missing_before = df_filled[col].isnull().sum()
        if missing_before == 0:
            continue
            
        # Calcola la moda all'interno del gruppo
        group_mode = df_filled.groupby(group_col)[col].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )
        
        df_filled[col] = df_filled[col].fillna(group_mode)
        
        missing_after = df_filled[col].isnull().sum()
        imputed_count = missing_before - missing_after
        print(f"Colonna '{col}': Imputati {imputed_count} valori usando il gruppo. Nulli rimanenti: {missing_after}")
        
    return df_filled

def impute_global_mode_fallback(df: pd.DataFrame, columns_to_impute: list) -> pd.DataFrame:
    """
    Fase 2: Riempie i valori nulli rimanenti (es. viaggiatori solitari) con la moda globale.
    """
    print(f"--- Avvio Imputazione Globale (Fallback) ---")
    df_filled = df.copy()
    
    for col in columns_to_impute:
        missing_before = df_filled[col].isnull().sum()
        if missing_before > 0:
            global_mode = df_filled[col].mode().iloc[0]
            df_filled[col] = df_filled[col].fillna(global_mode)
            print(f"Colonna '{col}': Imputati {missing_before} valori usando la moda globale ('{global_mode}').")
            
    return df_filled

def run_handle_null_values(df: pd.DataFrame) -> HandleNullValueOutputs:
    """
    Esegue la pipeline di imputazione dei valori nulli per le colonne categoriali,
    incluse le componenti della Cabina (Deck, Num, Side).
    
    Args:
        df (pd.DataFrame): Il DataFrame originale con valori nulli.
        
    Returns:
        HandleNullValueOutputs: Il DataFrame imputato wrappato in una NamedTuple.
    """
    # Ho aggiunto 'Num' alla lista. 'Deck' e 'Side' erano già presenti.
    categorical_cols = ['HomePlanet', 'Destination', 'Deck', 'Num', 'Side', 'Surnames']
    
    # Filtriamo le colonne per assicurarci che esistano effettivamente nel DataFrame
    cols_to_impute = [col for col in categorical_cols if col in df.columns]
    
    # Esegui le due fasi
    df_step1 = impute_group_mode(df, columns_to_impute=cols_to_impute, group_col='Group')
    df_final = impute_global_mode_fallback(df_step1, columns_to_impute=cols_to_impute)
    
    print("Gestione valori nulli (OP4) completata con successo.")
    
    # Restituisci il risultato nello standard della pipeline
    return HandleNullValueOutputs(df_output=df_final)