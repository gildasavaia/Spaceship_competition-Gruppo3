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

    # Funzione interna per calcolare la moda di un gruppo
    def get_mode(x):
        m = x.dropna().mode()
        # Se il gruppo ha tutti NaN, la moda è vuota, quindi restituisce NaN
        return m.iloc[0] if not m.empty else np.nan

    # Calcola la moda per ogni gruppo e allinea i risultati tramite transform
    group_modes = df[multi_mask].groupby(group_col[multi_mask])[feature].transform(get_mode)
    
    # Applica i valori calcolati solo dove c'è il dato mancante
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

def run_handle_null_values(df_input: pd.DataFrame) -> HandleNullValuesResult:
    """
    Esegue l'imputazione dei valori mancanti:
    - Pulizia CryoSleep: Uniforma i booleani scritti come stringhe o numeri.
    - Age: usa la media totale.
    - Costi: sostituisce i NaN con 0.
    - Altre feature:
        - Gruppi con cardinalità > 1: usa la moda del gruppo.
        - Sezioni rimanenti (cardinalità = 1 OPPURE gruppi interamente vuoti): usa le probabilità globali.
    """
    df = df_input.copy()

    # --- INIZIO PULIZIA CRYOSLEEP ---
    if "CryoSleep" in df.columns:
        replace_dict = {
            '0.0': 'False', 
            '1.0': 'True',
            0.0: 'False',
            1.0: 'True',
            False: 'False',
            True: 'True'
        }
        df['CryoSleep'] = df['CryoSleep'].replace(replace_dict)
        print("[OP4] Pulizia feature 'CryoSleep': uniformati '0.0'/'1.0' in 'False'/'True'.\n")
    # --- FINE PULIZIA CRYOSLEEP ---

    # Sostituzione NaN con 0 per le feature dei costi
    cost_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    existing_cost_cols = [col for col in cost_cols if col in df.columns]
    if existing_cost_cols:
        n_missing_costs = df[existing_cost_cols].isna().sum().sum()
        df[existing_cost_cols] = df[existing_cost_cols].fillna(0)
        # print(f"[OP4] Feature dei costi ({', '.join(existing_cost_cols)}): riempiti {n_missing_costs} valori mancanti con 0.\n")

    # Imputazione colonna "Age" con la media totale 
    if "Age" in df.columns:
        n_missing_age = df["Age"].isna().sum()
        age_mean = int(df["Age"].mean() + 0.5)
        df["Age"] = df["Age"].fillna(age_mean)
        # print(f"[OP4] Feature 'Age': riempiti {n_missing_age} valori mancanti con la media totale ({age_mean}).\n")
    
    # Imputazione per le altre feature categoriali
    features = [
        "HomePlanet",
        "Destination",
        "CryoSleep",
        "VIP",
        "Deck",
        "Num",
        "Side"
    ]

    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        raise ValueError(
            f"Nel dataset mancano queste colonne richieste: {missing_features}\n"
            "Assicurati di aver precedentemente estratto Deck, Num e Side dalla colonna Cabin."
        )

    group_col = get_group_column(df)
    group_sizes = group_col.map(group_col.value_counts())
    
    singleton_mask = group_sizes == 1
    multi_mask = group_sizes > 1

    probability_dictionaries = {}
    for feature in features:
        probability_dictionaries[feature] = build_probability_dictionary(df[feature])

    # print("Dizionari delle probabilità (basati sull'intero dataset):\n")
    # for feature, prob_dict in probability_dictionaries.items():
    #     print(f"{feature}:")
    #     for key, value in prob_dict.items():
    #         print(f"  {key}: {value:.6f}")
    #     print()

    for i, feature in enumerate(features):
        # 1. Imputazione con la moda del gruppo (cardinalità > 1)
        before_multi = (df[feature].isna() & multi_mask).sum()
        df = impute_missing_group_mode(
            df=df,
            feature=feature,
            group_col=group_col,
            multi_mask=multi_mask
        )
        
        # Dopo la moda, controlliamo quanti valori sono rimasti NaN nella multi_mask
        # Questi sono esattamente i passeggeri appartenenti a gruppi in cui la feature era TUTTA NaN
        after_multi_remaining_nans = (df[feature].isna() & multi_mask).sum()
        filled_multi = before_multi - after_multi_remaining_nans

        # 2. Imputazione probabilistica
        # Uniamo la maschera dei singletons con i valori ancora NaN dei multi_groups
        prob_mask = singleton_mask | (df[feature].isna() & multi_mask)
        
        before_prob = (df[feature].isna() & prob_mask).sum()
        df = impute_missing_probabilistic(
            df=df,
            feature=feature,
            prob_dict=probability_dictionaries[feature],
            target_mask=prob_mask,
            random_state=42 + i
        )
        after_prob = (df[feature].isna() & prob_mask).sum()
        filled_prob = before_prob - after_prob

        # print(
        #     f"[OP4] Feature '{feature}': "
        #     f"riempiti {filled_multi} valori (moda gruppi multipli), "
        #     f"riempiti {filled_prob} valori (probabilità globale)."
        # )
        if after_multi_remaining_nans > 0:
            print(f"      -> Di questi {filled_prob} probabilistici, {after_multi_remaining_nans} provenivano da gruppi multipli completamente vuoti.")

    # print(f"\n[OP4] Statistiche gruppi:")
    # print(f"  - Passeggeri in gruppi multipli (>1): {multi_mask.sum()}")
    # print(f"  - Passeggeri in gruppi solitari (=1): {singleton_mask.sum()}")

    return HandleNullValuesResult(
        df_output=df,
        probability_dictionaries=probability_dictionaries
    )

def main():
    input_path = "train_processed.csv"  # Modificato in base al tuo file
    output_path = "train_imputed_singletons.csv"

    # Creo un dataframe fittizio di test per dimostrazione se non trova il file, 
    # ma il comportamento principale usa pd.read_csv()
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Attenzione: File '{input_path}' non trovato. Assicurati che esista nella directory.")
        return

    risultato = run_handle_null_values(df)

    risultato.df_output.to_csv(output_path, index=False)
    print(f"\nDataset salvato in: {output_path}")

if __name__ == "__main__":
    main()