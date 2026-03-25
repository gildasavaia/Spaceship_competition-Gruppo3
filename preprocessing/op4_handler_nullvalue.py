import pandas as pd
import numpy as np
from typing import NamedTuple

# Creiamo una NamedTuple per l'output, seguendo lo standard della tua pipeline
class HandleNullValueOutputs(NamedTuple):
    df_output: pd.DataFrame


DESTINATION_VALUES = ['TRAPPIST-1e', '55 Cancri e', 'PSO J318.5-22']
DESTINATION_WEIGHTS = np.array([0.6985, 0.2089, 0.0926], dtype=float)

def impute_group_mode(df: pd.DataFrame, columns_to_impute: list, group_col: str = 'Group') -> pd.DataFrame:
    """
    Fase 1: Imputa i valori nulli secondo la strategia:
    - Per viaggiatori in gruppo (size > 1): moda del loro gruppo
    - Per viaggiatori solitari (size = 1): moda globale del dataset
    """
    print(f"\n--- Avvio Imputazione CATEGORIALE basata su Gruppo/Solitario ({group_col}) ---")
    df_filled = df.copy()
    rng = np.random.default_rng(42)
    
    # Identifica i viaggiatori solitari (gruppo di size 1)
    group_sizes = df_filled.groupby(group_col).size()
    is_solo_traveler = df_filled[group_col].map(group_sizes) == 1
    
    for col in columns_to_impute:
        missing_before = df_filled[col].isnull().sum()
        if missing_before == 0:
            continue
        
        # Calcola moda globale per il dataset
        global_mode = df_filled[col].mode()
        global_mode = global_mode.iloc[0] if not global_mode.empty else np.nan
        
        # Calcola la moda all'interno di ogni gruppo
        group_mode = df_filled.groupby(group_col)[col].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )
        
        # Identifica le righe con valori nulli
        mask_null = df_filled[col].isnull()
        
        # Conteggia quanti solitari e gruppi hanno null PRIMA del riempimento
        solo_mask = mask_null & is_solo_traveler
        group_mask = mask_null & ~is_solo_traveler
        
        # Applica strategia ai solitari con null
        if col == 'Destination':
            solo_count = int(solo_mask.sum())
            if solo_count > 0:
                random_destinations = rng.choice(
                    DESTINATION_VALUES,
                    size=solo_count,
                    p=DESTINATION_WEIGHTS / DESTINATION_WEIGHTS.sum()
                )
                # da mettere in funzione della sommma dei 3 elementi e fare tutte le altre feauture in questo modo
                df_filled.loc[solo_mask, col] = random_destinations
        else:
            df_filled.loc[solo_mask, col] = global_mode
        
        # Applica moda di gruppo agli altri con null
        df_filled.loc[group_mask, col] = group_mode[group_mask]
        
        missing_after = df_filled[col].isnull().sum()
        imputed_count = missing_before - missing_after
        
        # Conteggia quanti sono stati EFFETTIVAMENTE imputati per categoria
        solo_imputed = solo_mask.sum() - (df_filled[solo_mask][col].isnull().sum() if solo_mask.sum() > 0 else 0)
        group_imputed = group_mask.sum() - (df_filled[group_mask][col].isnull().sum() if group_mask.sum() > 0 else 0)
        
        if col == 'Destination':
            solo_strategy = 'estrazione casuale pesata'
        else:
            solo_strategy = 'moda globale'

        print(f"Colonna '{col}': Imputati {imputed_count} valori ({solo_imputed} solitari con {solo_strategy}, {group_imputed} in gruppo con moda di gruppo). Nulli rimanenti: {missing_after}")
        
    return df_filled

def impute_group_mean(df: pd.DataFrame, columns_to_impute: list, group_col: str = 'Group') -> pd.DataFrame:
    """
    Imputa i valori nulli per colonne NUMERICHE secondo la strategia:
    - Per viaggiatori in gruppo (size > 1): media del loro gruppo
    - Per viaggiatori solitari (size = 1): media globale del dataset
    """
    print(f"\n--- Avvio Imputazione NUMERICA basata su Gruppo/Solitario ({group_col}) ---")
    df_filled = df.copy()
    
    # Identifica i viaggiatori solitari (gruppo di size 1)
    group_sizes = df_filled.groupby(group_col).size()
    is_solo_traveler = df_filled[group_col].map(group_sizes) == 1
    
    for col in columns_to_impute:
        missing_before = df_filled[col].isnull().sum()
        if missing_before == 0:
            continue
        
        # Calcola media globale per il dataset
        global_mean = df_filled[col].mean()
        
        # Calcola la media all'interno di ogni gruppo
        group_mean = df_filled.groupby(group_col)[col].transform('mean')
        
        # Identifica le righe con valori nulli
        mask_null = df_filled[col].isnull()
        
        # Conteggia quanti solitari e gruppi hanno null PRIMA del riempimento
        solo_mask = mask_null & is_solo_traveler
        group_mask = mask_null & ~is_solo_traveler
        
        # Applica media globale ai solitari con null
        df_filled.loc[solo_mask, col] = global_mean
        
        # Applica media di gruppo agli altri con null
        df_filled.loc[group_mask, col] = group_mean[group_mask]
        
        missing_after = df_filled[col].isnull().sum()
        imputed_count = missing_before - missing_after
        
        # Conteggia quanti sono stati EFFETTIVAMENTE imputati per categoria
        solo_imputed = solo_mask.sum() - (df_filled[solo_mask][col].isnull().sum() if solo_mask.sum() > 0 else 0)
        group_imputed = group_mask.sum() - (df_filled[group_mask][col].isnull().sum() if group_mask.sum() > 0 else 0)
        
        print(f"Colonna '{col}': Imputati {imputed_count} valori ({solo_imputed} solitari con media globale, {group_imputed} in gruppo con media di gruppo). Nulli rimanenti: {missing_after}")
        
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
    Esegue la pipeline di imputazione dei valori nulli per le colonne categoriali e numeriche.
    
    Strategia CATEGORIALE (moda):
    - Per viaggiatori SOLITARI (gruppo size=1) usa la moda globale del dataset.
    - Per viaggiatori IN GRUPPO (size>1) usa la moda del loro gruppo.
    
    Strategia NUMERICA (media):
    - Per viaggiatori SOLITARI (gruppo size=1) usa la media globale del dataset.
    - Per viaggiatori IN GRUPPO (size>1) usa la media del loro gruppo.
    
    Fallback: Riempie null residui con media/moda globale.
    
    Args:
        df (pd.DataFrame): Il DataFrame originale con valori nulli.
        
    Returns:
        HandleNullValueOutputs: Il DataFrame imputato wrappato in una NamedTuple.
    """
    # Colonne categoriali: usano la moda
    categorical_cols = ['HomePlanet', 'Destination', 'Deck', 'Num', 'Side', 'Surnames']
    
    # Colonne numeriche: usano la media
    numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # Filtriamo le colonne per assicurarci che esistano effettivamente nel DataFrame
    cat_cols_to_impute = [col for col in categorical_cols if col in df.columns]
    num_cols_to_impute = [col for col in numeric_cols if col in df.columns]
    
    # Esegui imputazione categoriale
    df_step1 = impute_group_mode(df, columns_to_impute=cat_cols_to_impute, group_col='Group')
    
    # Esegui imputazione numerica
    df_step2 = impute_group_mean(df_step1, columns_to_impute=num_cols_to_impute, group_col='Group')
    
    # Fallback per categorie
    df_final = impute_global_mode_fallback(df_step2, columns_to_impute=cat_cols_to_impute)
    
    print("\nGestione valori nulli (OP4) completata con successo.")
    
    # Restituisci il risultato nello standard della pipeline
    return HandleNullValueOutputs(df_output=df_final)