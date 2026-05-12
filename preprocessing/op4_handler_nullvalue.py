import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class HandleNullValuesResult:
    df_output: pd.DataFrame
    probability_dictionaries: dict
    age_mean: float = None

def build_probability_dictionary(series: pd.Series) -> dict:
    """
    Costruisce un dizionario:
    chiave = valore osservato nella feature
    valore = frequenza relativa del valore nella colonna

    I NaN vengono ignorati.
    """
    freq = series.dropna().value_counts(normalize=True)
    return freq.to_dict()

def get_group_column(df: pd.DataFrame) -> pd.Series:
    """
    Restituisce la colonna che identifica il gruppo.

    Priorità:
    1. usa 'Group' se esiste
    2. altrimenti la ricava da 'PassengerId' prendendo la parte prima di '_'
    """
    if "Group" in df.columns:
        return df["Group"]

    if "PassengerId" in df.columns:
        return df["PassengerId"].astype(str).str.split("_").str[0]

    raise ValueError(
        "Non trovo né la colonna 'Group' né la colonna 'PassengerId' per identificare i gruppi."
    )

def impute_missing_group_mode(
    df: pd.DataFrame,
    feature: str,
    group_col: pd.Series,
    multi_mask: pd.Series
) -> pd.DataFrame:
    """
    Riempie i valori mancanti della feature SOLO per i passeggeri
    che appartengono a gruppi di cardinalità > 1, utilizzando la moda del gruppo.
    """
    mask_to_fill = df[feature].isna() & multi_mask
    if mask_to_fill.sum() == 0:
        return df

    def get_mode(x):
        m = x.dropna().mode()
        return m.iloc[0] if not m.empty else np.nan

    group_modes = df[multi_mask].groupby(group_col[multi_mask])[feature].transform(get_mode)
    df.loc[mask_to_fill, feature] = group_modes[mask_to_fill]
    
    return df

def impute_missing_probabilistic(
    df: pd.DataFrame,
    feature: str,
    prob_dict: dict,
    target_mask: pd.Series,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Riempie i valori mancanti della feature estraendo valori in base alle probabilità globali.
    Si applica ai passeggeri identificati da target_mask (es. singletons + gruppi interamente NaN).
    """
    if not prob_dict:
        return df

    rng = np.random.default_rng(random_state)

    mask_to_fill = df[feature].isna() & target_mask
    n_missing = mask_to_fill.sum()

    if n_missing == 0:
        return df

    values = list(prob_dict.keys())
    probs = list(prob_dict.values())

    sampled_values = rng.choice(values, size=n_missing, p=probs)

    df.loc[mask_to_fill, feature] = sampled_values
    return df

def enforce_cryo_sleep_spending_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Se CryoSleep == True → tutte le spese devono essere 0.
    Corregge eventuali inconsistenze presenti nel dataset.
    """
    cost_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpending']
    existing_cost_cols = [col for col in cost_cols if col in df.columns]

    if "CryoSleep" not in df.columns or not existing_cost_cols:
        return df

    total_spending = df[existing_cost_cols].sum(axis=1)

    mask = (df['CryoSleep'] == 'True') & (total_spending > 0)
    n_fixed = mask.sum()

    if n_fixed > 0:
        df.loc[mask, existing_cost_cols] = 0
        print(f"[OP4] Corrette {n_fixed} righe: CryoSleep=True ma spesa > 0 → impostata a 0.")
        
    if "Deck" not in df.columns or "VIP" not in df.columns:
        return df

    # Troviamo i passeggeri che sono sui ponti G o T e hanno CryoSleep mancante
    mask_to_fix = df['Deck'].isin(['T']) & df['CryoSleep'].isna()
    n_fixed = mask_to_fix.sum()

    if n_fixed > 0:
        # Impostiamo a 'False' (usiamo la stringa come nel resto del tuo OP4)
        df.loc[mask_to_fix, 'CryoSleep'] = 'False'
        print(f"[OP4] Regola CryoSleep: {n_fixed} passeggeri sui ponti G o T imputati come CryoSleep='False'.")
        
    mask_nan = df['CryoSleep'].isna() & (total_spending > 0)
    n_filled = mask_nan.sum()

    if n_filled > 0:
        df.loc[mask_nan, 'CryoSleep'] = 'False'
        print(f"[OP4] Riempiti {n_filled} NaN in CryoSleep con 'False' poiché la spesa totale > 0.")
        
    return df

def enforce_children_spending_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regola logica: I passeggeri con meno di 13 anni (Age < 13) non hanno crediti.
    Forza tutte le colonne di costo a 0 per i bambini.
    """
    cost_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpending']
    existing_cost_cols = [col for col in cost_cols if col in df.columns]

    if "Age" not in df.columns or not existing_cost_cols:
        return df

    # Creiamo una maschera per trovare chi ha meno di 13 anni
    mask_children = df['Age'] < 13
    
    n_children = mask_children.sum()

    if n_children > 0:
        # Impostiamo a 0 tutte le colonne di costo per i bambini
        df.loc[mask_children, existing_cost_cols] = 0
        print(f"[OP4] Regola Età: Forzati a 0 i costi per {n_children} bambini (Età < 13).")

    return df

def enforce_vip_age_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regola logica: I passeggeri minorenni (Age < 18) non sono MAI VIP.
    Forza il valore VIP a 'False' per chi ha meno di 18 anni.
    """
    if "Age" not in df.columns or "VIP" not in df.columns:
        return df

    # Troviamo i passeggeri minorenni con VIP mancante (o settato per sbaglio a True)
    mask_under_18_nan = (df['Age'] < 18) & df['VIP'].isna()
    n_fixed_nan = mask_under_18_nan.sum()

    if n_fixed_nan > 0:
        df.loc[mask_under_18_nan, 'VIP'] = 'False'
        print(f"[OP4] Regola Età/VIP: {n_fixed_nan} passeggeri minorenni imputati come VIP='False'.")
        
    return df

def enforce_vip_deck_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regola logica: I passeggeri nei ponti G e T non sono MAI VIP.
    Forza il valore VIP a 'False' per chi alloggia in questi ponti.
    I passseggeri con HomePlanet=Earth non sono VIP.
    """
    if "Deck" not in df.columns or "VIP" not in df.columns:
        return df

    # Troviamo i passeggeri che sono sui ponti G o T e hanno VIP mancante
    mask_to_fix = df['Deck'].isin(['G', 'T']) & df['VIP'].isna()
    n_fixed = mask_to_fix.sum()

    if n_fixed > 0:
        # Impostiamo a 'False' (usiamo la stringa come nel resto del tuo OP4)
        df.loc[mask_to_fix, 'VIP'] = 'False'
        print(f"[OP4] Regola VIP: {n_fixed} passeggeri sui ponti G o T imputati come VIP='False'.")
        
    if "HomePlanet" in df.columns and "VIP" in df.columns:
        mask = (df["HomePlanet"] == "Earth") & (df["VIP"].isna())
        n_fixed = mask.sum()
        
        if n_fixed > 0:
            df.loc[mask, "VIP"] = "False"
            print(f"[OP4] Regola Terra-VIP: {n_fixed} passeggeri terrestri con VIP NaN impostati a 'False'.")


    return df

def enforce_deck_homeplanet_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Se `HomePlanet` è NaN ma `Deck` è tra A, B, C, T,
    imposta `HomePlanet` = 'Europa'. Questo applica la regola
    logica richiesta per alcune cabine/ponte.
    """
    mask_europa = df['Deck'].isin(['A', 'B', 'C', 'T']) & df['HomePlanet'].isna()
    df.loc[mask_europa, 'HomePlanet'] = 'Europa'
    
    # Regola 2: Ponte G -> Earth
    mask_earth = (df['Deck'] == 'G') & df['HomePlanet'].isna()
    df.loc[mask_earth, 'HomePlanet'] = 'Earth'

    return df

def run_handle_null_values(df_input: pd.DataFrame,  train_prob_dicts=None, train_age_mean=None,) -> HandleNullValuesResult:
    """
    Esegue l'imputazione dei valori mancanti:
    - Pulizia CryoSleep e VIP
    - Age: media
    - Costi: NaN → 0
    - Regola logica CryoSleep
    - Altre feature: moda gruppo + probabilistico
    """
    df = df_input.copy()

    # --- INIZIO PULIZIA BOOLEANE (CRYOSLEEP E VIP) ---
    bool_cols = ["CryoSleep", "VIP"]
    replace_dict = {
        '0.0': 'False', 
        '1.0': 'True',
        0.0: 'False',
        1.0: 'True',
        False: 'False',
        True: 'True'
    }
    
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].replace(replace_dict)
            # Fondamentale per evitare la divisione in float/stringhe in fase di encoding
            df.loc[df[col].notna(), col] = df.loc[df[col].notna(), col].astype(str)
            print(f"[OP4] Pulizia feature '{col}'\n")

    

    # 🔥 REGOLA LOGICA
    df = enforce_cryo_sleep_spending_rule(df)
    df = enforce_vip_age_rule(df)
    df = enforce_vip_deck_rule(df)
    df = enforce_children_spending_rule(df) 
     
     
     # COSTI
    cost_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    existing_cost_cols = [col for col in cost_cols if col in df.columns]

    if existing_cost_cols:
        n_missing_costs = df[existing_cost_cols].isna().sum().sum()
        df[existing_cost_cols] = df[existing_cost_cols].fillna(0)
        print(f"[OP4] Costi: {n_missing_costs} NaN → 0\n")
        
    # AGE
    
    age_mean_to_return = train_age_mean
    
    if "Age" in df.columns:
        n_missing_age = df["Age"].isna().sum()
        # Se ci viene passata la media del train (siamo nel Test) usiamo quella, altrimenti la calcoliamo (siamo nel Train)
        age_mean_to_use = train_age_mean if train_age_mean is not None else int(df["Age"].mean() + 0.5)
        df["Age"] = df["Age"].fillna(age_mean_to_use)
        age_mean_to_return = age_mean_to_use

    # Regola custom: se HomePlanet è NaN ma il ponte è A/B/C/T → HomePlanet='Europa'
    df = enforce_deck_homeplanet_rule(df)
    
    features = [
        "HomePlanet", "Destination", "CryoSleep",
        "VIP", "Deck", "Num", "Side", 'Surnames'
    ]




    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Mancano colonne: {missing_features}")

    group_col = get_group_column(df)
    group_sizes = group_col.map(group_col.value_counts())

    singleton_mask = group_sizes == 1
    multi_mask = group_sizes > 1

    if train_prob_dicts is None:
        probability_dictionaries = {}
        for feature in features:
            probability_dictionaries[feature] = build_probability_dictionary(df[feature])

    else:
        probability_dictionaries = train_prob_dicts
        
    for i, feature in enumerate(features):

        # --- MODIFICA: Regola speciale per 'Destination' ---
        if feature == "Destination":
            # Per Destination NON usiamo la moda del gruppo (coesione troppo bassa, ~47%)
            filled_multi = 0
            
            # La maschera di probabilità copre TUTTI i valori mancanti (sia single che gruppi)
            prob_mask = df[feature].isna()
            
            before_prob = (df[feature].isna() & prob_mask).sum()

            df = impute_missing_probabilistic(
                df,
                feature,
                probability_dictionaries[feature],
                prob_mask,
                random_state=42 + i
            )

            after_prob = (df[feature].isna() & prob_mask).sum()
            filled_prob = before_prob - after_prob
            
        else:
            # --- COMPORTAMENTO STANDARD PER LE ALTRE FEATURE ---
            before_multi = (df[feature].isna() & multi_mask).sum()

            df = impute_missing_group_mode(
                df, feature, group_col, multi_mask
            )

            after_multi = (df[feature].isna() & multi_mask).sum()
            filled_multi = before_multi - after_multi

            prob_mask = singleton_mask | (df[feature].isna() & multi_mask)

            before_prob = (df[feature].isna() & prob_mask).sum()

            df = impute_missing_probabilistic(
                df,
                feature,
                probability_dictionaries[feature],
                prob_mask,
                random_state=42 + i
            )

            after_prob = (df[feature].isna() & prob_mask).sum()
            filled_prob = before_prob - after_prob

        # Stampa i risultati unificati
        print(
            f"[OP4] {feature}: "
            f"{filled_multi} (moda), "
            f"{filled_prob} (prob)"
        )

    # Corretto
    return HandleNullValuesResult(df, probability_dictionaries, age_mean_to_return)

# def main():
#     input_path = "train_processed.csv"
#     output_path = "train_imputed_singletons.csv"

#     try:
#         df = pd.read_csv(input_path)
#     except FileNotFoundError:
#         print(f"File non trovato: {input_path}")
#         return

#     result = run_handle_null_values(df)

#     result.df_output.to_csv(output_path, index=False)
#     print(f"\nSalvato in: {output_path}")

# if __name__ == "__main__":
#     main()