import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class HandleNullValuesResult:
    df_output: pd.DataFrame
    probability_dictionaries: dict

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
    cost_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    existing_cost_cols = [col for col in cost_cols if col in df.columns]

    if "CryoSleep" not in df.columns or not existing_cost_cols:
        return df

    total_spending = df[existing_cost_cols].sum(axis=1)

    mask = (df['CryoSleep'] == 'True') & (total_spending > 0)
    n_fixed = mask.sum()

    if n_fixed > 0:
        df.loc[mask, existing_cost_cols] = 0
        print(f"[OP4] Corrette {n_fixed} righe: CryoSleep=True ma spesa > 0 → impostata a 0.")

    return df

def run_handle_null_values(df_input: pd.DataFrame) -> HandleNullValuesResult:
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
            df[col] = df[col].astype(str) 
            print(f"[OP4] Pulizia feature '{col}'\n")

    # COSTI
    cost_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    existing_cost_cols = [col for col in cost_cols if col in df.columns]

    if existing_cost_cols:
        n_missing_costs = df[existing_cost_cols].isna().sum().sum()
        df[existing_cost_cols] = df[existing_cost_cols].fillna(0)
        print(f"[OP4] Costi: {n_missing_costs} NaN → 0\n")

    # 🔥 REGOLA LOGICA
    df = enforce_cryo_sleep_spending_rule(df)

    # AGE
    if "Age" in df.columns:
        n_missing_age = df["Age"].isna().sum()
        age_mean = int(df["Age"].mean() + 0.5)
        df["Age"] = df["Age"].fillna(age_mean)
        print(f"[OP4] Age: {n_missing_age} NaN → {age_mean}\n")
    
    features = [
        "HomePlanet", "Destination", "CryoSleep",
        "VIP", "Deck", "Num", "Side"
    ]

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Mancano colonne: {missing_features}")

    group_col = get_group_column(df)
    group_sizes = group_col.map(group_col.value_counts())

    singleton_mask = group_sizes == 1
    multi_mask = group_sizes > 1

    probability_dictionaries = {}
    for feature in features:
        probability_dictionaries[feature] = build_probability_dictionary(df[feature])

    for i, feature in enumerate(features):

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

        print(
            f"[OP4] {feature}: "
            f"{filled_multi} (moda), "
            f"{filled_prob} (prob)"
        )

    return HandleNullValuesResult(df, probability_dictionaries)

def main():
    input_path = "train_processed.csv"
    output_path = "train_imputed_singletons.csv"

    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"File non trovato: {input_path}")
        return

    result = run_handle_null_values(df)

    result.df_output.to_csv(output_path, index=False)
    print(f"\nSalvato in: {output_path}")

if __name__ == "__main__":
    main()