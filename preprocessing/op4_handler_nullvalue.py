import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder

@dataclass
class HandleNullValuesResult:
    df_output: pd.DataFrame
    probability_dictionaries: dict
    age_global_median: float = None
    age_homeplanet_medians: dict = field(default_factory=dict)  # Mediana Age per HomePlanet (dal train)
    age_deck_medians: dict = field(default_factory=dict)        # Mediana Age per Deck (dal train)
    numzone_model: object = None        # RandomForestClassifier fittato su train per NumZone
    numzone_encoder: object = None      # OrdinalEncoder per le feature predittorie

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


def impute_numzone_multivariate(
    df: pd.DataFrame,
    group_col: pd.Series,
    multi_mask: pd.Series,
    model=None,
    encoder=None,
    random_state: int = 42
):
    """
    Imputa NumZone con strategia ibrida:
    - Moda del gruppo per i passeggeri in gruppi > 1 (se almeno un membro ha NumZone nota)
    - RandomForestClassifier per i singleton (GroupSize == 1) e per i gruppi
      dove tutti i membri hanno NumZone NaN

    Feature predittorie: ['Deck', 'Side', 'HomePlanet', 'Destination', 'VIP']

    Se `model` è None si è in fase di TRAIN: il modello viene fittato sui dati
    con NumZone nota e restituito assieme all'encoder.
    Se `model` è passato si è in fase di TEST: il modello viene usato solo per predict.

    Returns:
        df, model, encoder
    """
    PREDICTOR_COLS = ['Deck', 'Side', 'HomePlanet', 'Destination', 'VIP']

    # --- Step 1: moda del gruppo (multi_mask) ---
    df = impute_missing_group_mode(df, 'NumZone', group_col, multi_mask)
    filled_group = df['NumZone'].notna().sum()

    # --- Step 2: RandomForest per i NaN rimanenti ---
    mask_still_nan = df['NumZone'].isna()
    n_still_nan = mask_still_nan.sum()

    if n_still_nan == 0:
        print("[OP4] NumZone: tutti i NaN riempiti con la moda del gruppo. Nessun intervento RF necessario.")
        return df, model, encoder

    # Verifica che le colonne predittorie esistano
    missing_preds = [c for c in PREDICTOR_COLS if c not in df.columns]
    if missing_preds:
        raise ValueError(f"[OP4] NumZone multivariate: mancano colonne predittorie {missing_preds}")

    if model is None:
        # --- TRAIN: fit del modello ---
        train_mask = df['NumZone'].notna()
        # Selezioniamo solo le righe con tutte le feature predittorie non-null
        predictor_notna_mask = df[PREDICTOR_COLS].notna().all(axis=1)
        fit_mask = train_mask & predictor_notna_mask

        X_fit_raw = df.loc[fit_mask, PREDICTOR_COLS]
        # NumZone è ora float (1.0–10.0): convertiamo in int per il classificatore
        y_fit     = df.loc[fit_mask, 'NumZone'].astype(float).astype(int)

        # Encoding ordinale delle feature categoriche
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_fit_enc = encoder.fit_transform(X_fit_raw)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_fit_enc, y_fit)
        print(f"[OP4] NumZone RF: modello fittato su {fit_mask.sum()} righe (feature: {PREDICTOR_COLS}).")
    else:
        print(f"[OP4] NumZone RF: uso modello del train per predire {n_still_nan} NaN.")

    # --- Predict sui NaN rimanenti con feature predittorie note ---
    predict_notna_mask = df[PREDICTOR_COLS].notna().all(axis=1)
    predict_mask = mask_still_nan & predict_notna_mask
    n_predict = predict_mask.sum()

    if n_predict > 0:
        X_pred_raw = df.loc[predict_mask, PREDICTOR_COLS]
        X_pred_enc = encoder.transform(X_pred_raw)
        y_pred     = model.predict(X_pred_enc)
        # Riconvertiamo le predizioni (int) in float per coerenza con il tipo di NumZone
        df.loc[predict_mask, 'NumZone'] = pd.to_numeric(y_pred, errors='coerce').astype(float)
        print(f"[OP4] NumZone RF: predetti {n_predict} valori.")

    # Eventuali NaN rimasti (es. predittori anch'essi NaN): fallback alla moda globale
    still_nan_after_rf = df['NumZone'].isna().sum()
    if still_nan_after_rf > 0:
        global_mode = df['NumZone'].mode().iloc[0]
        df['NumZone'] = df['NumZone'].fillna(global_mode)
        print(f"[OP4] NumZone: {still_nan_after_rf} NaN residui → moda globale ({global_mode}).")

    return df, model, encoder

def enforce_cryo_sleep_spending_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Se CryoSleep è NaN ma la spesa totale > 0 → imposta CryoSleep a 'False'.
    Se CryoSleep è NaN ma il passeggero è sui ponti G o T → imposta CryoSleep a 'False'.
    Se CryoSleep è NaN ma TotalSpending è 0 → imposta CryoSleep a 'True'.
    """
    cost_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpending']
    existing_cost_cols = [col for col in cost_cols if col in df.columns]

    if "CryoSleep" not in df.columns or not existing_cost_cols:
        return df

    total_spending = df[existing_cost_cols].sum(axis=1)
    
    # Se spending=0 e CryoSleep è NaN → imputa CryoSleep='True'
    mask_zero = df['CryoSleep'].isna() & (total_spending == 0)
    n_fixed_zero = mask_zero.sum()
    if n_fixed_zero > 0:
        df.loc[mask_zero, 'CryoSleep'] = 'True'
        print(f"[OP4] Regola CryoSleep: {n_fixed_zero} passeggeri con spesa 0 imputati come CryoSleep='True'.")
        
    if "Deck" not in df.columns or "VIP" not in df.columns:
        return df

    # Troviamo i passeggeri che sono sui ponti G o T e hanno CryoSleep mancante
    mask_to_fix = df['Deck'].isin(['T', 'G']) & df['CryoSleep'].isna()
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
    Regola logica: Per i minorenni (Age < 13) impostiamo a 0 i valori mancanti 
    nelle colonne di spesa.
    """
    cost_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpending']
    existing_cost_cols = [col for col in cost_cols if col in df.columns]

    if "Age" not in df.columns or not existing_cost_cols:
        return df

    # Creiamo una maschera per trovare chi ha meno di 13 anni
    mask_under_13 = df['Age'] < 13
    
    for col in existing_cost_cols:
        mask_nan = df[col].isna()
        mask_to_fill = mask_under_13 & mask_nan
        n_filled = mask_to_fill.sum()
        if n_filled > 0:
            df.loc[mask_to_fill, col] = 0
            print(f"[OP4] Regola Età: Riempiti {n_filled} NaN in {col} con 0 per passeggeri con Age < 13.")

    return df

def enforce_vip_age_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regola logica: I passeggeri minorenni (Age < 18) non sono MAI VIP.
    """
    if "Age" not in df.columns or "VIP" not in df.columns:
        return df

    # Troviamo i passeggeri minorenni con VIP mancante
    mask_under_18_nan = (df['Age'] < 18) & df['VIP'].isna()
    n_fixed_nan = mask_under_18_nan.sum()

    if n_fixed_nan > 0:
        df.loc[mask_under_18_nan, 'VIP'] = 'False'
        print(f"[OP4] Regola Età/VIP: {n_fixed_nan} passeggeri minorenni non sono VIP.")
        
    return df

def enforce_vip_deck_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regola logica: I passeggeri nei ponti G e T non sono MAI VIP.
    Forza il valore VIP a 'False' per chi alloggia in questi ponti.
    I passeggeri con 'HomePlanet' = 'Earth' non sono VIP.
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
    Se `HomePlanet` è NaN ma `Deck` è tra A, B, C, T, imposta `HomePlanet` = 'Europa'. 
    Se `HomePlanet` è NaN e `Deck` è D e `VIP` è `True`, imposta `HomePlanet` = 'Europa'.
    Se 'Homeplanet' è NaN e 'Deck' è 'G', imposta 'Homeplanet' = 'Earth'.
    """
    mask_europa = df['Deck'].isin(['A', 'B', 'C', 'T']) & df['HomePlanet'].isna()
    df.loc[mask_europa, 'HomePlanet'] = 'Europa'
    
    # Regola 2: Ponte G -> Earth
    mask_earth = (df['Deck'] == 'G') & df['HomePlanet'].isna()
    df.loc[mask_earth, 'HomePlanet'] = 'Earth'
    
    # Regola 3: Ponte D e VIP -> Europa
    if 'VIP' in df.columns:
        mask_d_vip = (df['Deck'] == 'D') & (df['VIP'] == 'True') & df['HomePlanet'].isna()
        n_fixed_vip = mask_d_vip.sum()
        if n_fixed_vip > 0:
            df.loc[mask_d_vip, 'HomePlanet'] = 'Europa'
            print(f"[OP4] Regola HomePlanet: {n_fixed_vip} passeggeri VIP in Deck D imputati come HomePlanet='Europa'.")

    return df

def impute_age(
    df: pd.DataFrame,
    group_col: pd.Series,
    train_age_global_median: float = None,
    train_age_homeplanet_medians: dict = None,
    train_age_deck_medians: dict = None,
):
    """
    Imputa Age con strategia a cascata (nessun data leakage: mediane calcolate solo sul train):
    1. Mediana del gruppo (per passeggeri con almeno un compagno con Age nota)
    2. Mediana per HomePlanet (calcolata sul train)
    3. Mediana per Deck (calcolata sul train)
    4. Mediana globale del train (fallback finale)

    Returns:
        df, age_global_median, age_homeplanet_medians, age_deck_medians
    """
    if "Age" not in df.columns:
        return df, None, {}, {}

    # --- Calcola o riusa le statistiche del train ---
    if train_age_global_median is None:
        # TRAIN: calcoliamo le mediane sui dati con Age nota
        age_homeplanet_medians = (
            df.groupby("HomePlanet")["Age"].median().dropna().to_dict()
            if "HomePlanet" in df.columns else {}
        )
        age_deck_medians = (
            df.groupby("Deck")["Age"].median().dropna().to_dict()
            if "Deck" in df.columns else {}
        )
        age_global_median = df["Age"].median()
    else:
        # TEST: usiamo le statistiche del train per evitare data leakage
        age_homeplanet_medians = train_age_homeplanet_medians or {}
        age_deck_medians       = train_age_deck_medians or {}
        age_global_median      = train_age_global_median

    n_missing_total = df["Age"].isna().sum()
    if n_missing_total == 0:
        return df, age_global_median, age_homeplanet_medians, age_deck_medians

    # --- Step 1: mediana del gruppo ---
    group_age_median = df.groupby(group_col)["Age"].transform("median")
    mask_step1 = df["Age"].isna() & group_age_median.notna()
    n_step1 = int(mask_step1.sum())
    if n_step1 > 0:
        df.loc[mask_step1, "Age"] = group_age_median[mask_step1]

    # --- Step 2: mediana per HomePlanet ---
    n_step2 = 0
    if "HomePlanet" in df.columns and age_homeplanet_medians:
        mask_step2 = df["Age"].isna() & df["HomePlanet"].notna()
        n_step2 = int(mask_step2.sum())
        if n_step2 > 0:
            df.loc[mask_step2, "Age"] = df.loc[mask_step2, "HomePlanet"].map(age_homeplanet_medians)

    # --- Step 3: mediana per Deck ---
    n_step3 = 0
    if "Deck" in df.columns and age_deck_medians:
        mask_step3 = df["Age"].isna() & df["Deck"].notna()
        n_step3 = int(mask_step3.sum())
        if n_step3 > 0:
            df.loc[mask_step3, "Age"] = df.loc[mask_step3, "Deck"].map(age_deck_medians)

    # --- Step 4: mediana globale (fallback finale) ---
    n_step4 = int(df["Age"].isna().sum())
    if n_step4 > 0:
        df["Age"] = df["Age"].fillna(age_global_median)

    print(
        f"[OP4] Age: {n_missing_total} NaN imputati → "
        f"gruppo ({n_step1}), HomePlanet ({n_step2}), Deck ({n_step3}), globale ({n_step4})."
    )

    return df, age_global_median, age_homeplanet_medians, age_deck_medians


def run_handle_null_values(
    df_input: pd.DataFrame,
    train_prob_dicts=None,
    train_age_global_median=None,
    train_age_homeplanet_medians=None,
    train_age_deck_medians=None,
    train_numzone_model=None,
    train_numzone_encoder=None,
) -> HandleNullValuesResult:
    """
    Esegue l'imputazione dei valori mancanti:
    - Pulizia CryoSleep e VIP
    - Regole logiche (CryoSleep, VIP, HomePlanet)
    - Costi: NaN → 0
    - Age: mediana gruppo → HomePlanet → Deck → globale
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

    

    # REGOLA LOGICA
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

    # --- Group_col anticipato (serve per impute_age) ---
    group_col = get_group_column(df)
    group_sizes = group_col.map(group_col.value_counts())
    singleton_mask = group_sizes == 1
    multi_mask = group_sizes > 1

    # AGE: imputazione a cascata (gruppo → HomePlanet → Deck → mediana globale)
    df, age_global_median, age_homeplanet_medians, age_deck_medians = impute_age(
        df,
        group_col,
        train_age_global_median=train_age_global_median,
        train_age_homeplanet_medians=train_age_homeplanet_medians,
        train_age_deck_medians=train_age_deck_medians,
    )

    # Regola custom: se HomePlanet è NaN ma il ponte è A/B/C/T → HomePlanet='Europa'
    df = enforce_deck_homeplanet_rule(df)
    
    features = [
        "HomePlanet", "Destination", "CryoSleep",
        "VIP", "Deck", "Side"
        # NumZone è imputata separatamente con modello multivariato (vedi sotto)
    ]




    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Mancano colonne: {missing_features}")

    # group_col, group_sizes, singleton_mask, multi_mask già calcolati sopra

    # --- Imputazione multivariata di NumZone (prima del loop standard) ---
    df, numzone_model, numzone_encoder = impute_numzone_multivariate(
        df,
        group_col,
        multi_mask,
        model=train_numzone_model,
        encoder=train_numzone_encoder,
    )

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
        # print(
            # f"[OP4] {feature}: "
            # f"{filled_multi} (moda), "
            # f"{filled_prob} (prob)"
        # )

    return HandleNullValuesResult(
        df_output=df,
        probability_dictionaries=probability_dictionaries,
        age_global_median=age_global_median,
        age_homeplanet_medians=age_homeplanet_medians,
        age_deck_medians=age_deck_medians,
        numzone_model=numzone_model,
        numzone_encoder=numzone_encoder,
    )

