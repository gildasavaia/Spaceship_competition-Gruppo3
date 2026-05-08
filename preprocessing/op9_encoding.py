import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

class EncodingResult:
    def __init__(self, df_tree_train, df_nn_train, df_tree_test=None, df_nn_test=None):
        self.df_tree_train = df_tree_train
        self.df_nn_train = df_nn_train
        self.df_tree_test = df_tree_test
        self.df_nn_test = df_nn_test

def run_encoding(df_tree_train, df_nn_train, df_tree_test=None, df_nn_test=None):
    """
    Applica One-Hot Encoding per il ramo NN e Ordinal Encoding per il ramo Tree.
    Gestisce il fit sul train e transform sul test per evitare leakage.
    """
    
    # --- NOVITÀ: ELIMINAZIONE COLONNA 'Group' ---
    # Rimuoviamo la colonna Group in partenza da tutti i dataframe, se esiste
    # Usiamo errors='ignore' in modo che il codice non si blocchi se la colonna non c'è
    df_tree_train = df_tree_train.drop(columns=['Group'], errors='ignore')
    df_nn_train = df_nn_train.drop(columns=['Group'], errors='ignore')
    
    if df_tree_test is not None:
        df_tree_test = df_tree_test.drop(columns=['Group'], errors='ignore')
    if df_nn_test is not None:
        df_nn_test = df_nn_test.drop(columns=['Group'], errors='ignore')
    # ---------------------------------------------

    # Identifichiamo le colonne categoriche (escludendo il target se presente)
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    
    # Filtriamo solo quelle effettivamente presenti
    cat_cols = [c for c in cat_cols if c in df_tree_train.columns]

    # # --- 1. ORDINAL ENCODING (RAMO TREE) ---  # SE VUOI TESTARE ALBERI DECISIONALI CHE NON SIANO CAT BOOST O XGBOOST DECOMMENTA QUESTE RIGA
    # ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1) # SE VUOI TESTARE ALBERI DECISIONALI CHE NON SIANO CAT BOOST O XGBOOST DECOMMENTA QUESTE RIGA
    
    # df_tree_train_enc = df_tree_train.copy() # SE VUOI TESTARE ALBERI DECISIONALI CHE NON SIANO CAT BOOST O XGBOOST DECOMMENTA QUESTE RIGA
    # df_tree_train_enc[cat_cols] = ord_enc.fit_transform(df_tree_train[cat_cols].astype(str)) # SE VUOI TESTARE ALBERI DECISIONALI CHE NON SIANO CAT BOOST O XGBOOST DECOMMENTA QUESTE RIGA
    # df_tree_test_enc = None # SE VUOI TESTARE ALBERI DECISIONALI CHE NON SIANO CAT BOOST O XGBOOST DECOMMENTA QUESTE RIGA
    # if df_tree_test is not None: # SE VUOI TESTARE ALBERI DECISIONALI CHE NON SIANO CAT BOOST O XGBOOST DECOMMENTA QUESTE RIGA
    #     df_tree_test_enc = df_tree_test.copy() # SE VUOI TESTARE ALBERI DECISIONALI CHE NON SIANO CAT BOOST O XGBOOST DECOMMENTA QUESTE RIGA
    #     df_tree_test_enc[cat_cols] = ord_enc.transform(df_tree_test[cat_cols].astype(str)) # SE VUOI TESTARE ALBERI DECISIONALI CHE NON SIANO CAT BOOST O XGBOOST DECOMMENTA QUESTE RIGA

    # --- 2. ONE-HOT ENCODING (RAMO NN) ---
    oh_enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Fit e transform sul train
    oh_encoded_train = oh_enc.fit_transform(df_nn_train[cat_cols].astype(str))
    oh_cols = oh_enc.get_feature_names_out(cat_cols)
    
    # Creiamo il nuovo dataframe NN train
    df_nn_train_enc = pd.concat([
        df_nn_train.drop(columns=cat_cols).reset_index(drop=True),
        pd.DataFrame(oh_encoded_train, columns=oh_cols)
    ], axis=1)

    # --- NOVITÀ: ELIMINAZIONE COLONNA 'CryoSleep_False' NEL TRAIN ---
    df_nn_train_enc = df_nn_train_enc.drop(columns=['CryoSleep_False'], errors='ignore')
    df_nn_train_enc = df_nn_train_enc.drop(columns=['VIP_False'], errors='ignore')
    df_nn_train_enc = df_nn_train_enc.drop(columns=['Side_S'], errors='ignore')
    # ----------------------------------------------------------------

    # Transform sul test
    df_nn_test_enc = None
    if df_nn_test is not None:
        oh_encoded_test = oh_enc.transform(df_nn_test[cat_cols].astype(str))
        df_nn_test_enc = pd.concat([
            df_nn_test.drop(columns=cat_cols).reset_index(drop=True),
            pd.DataFrame(oh_encoded_test, columns=oh_cols)
        ], axis=1)
        
        # --- NOVITÀ: ELIMINAZIONE COLONNA 'CryoSleep_False' NEL TEST ---
        df_nn_test_enc = df_nn_test_enc.drop(columns=['CryoSleep_False'], errors='ignore')
        df_nn_test_enc = df_nn_test_enc.drop(columns=['VIP_False'], errors='ignore')
        df_nn_test_enc = df_nn_test_enc.drop(columns=['Side_S'], errors='ignore')
        # ---------------------------------------------------------------
    
    df_tree_train_enc = df_tree_train.copy()  # SE VUOI TESTARE ALBERI DECISIONALI CHE NON SIANO CAT BOOST O XGBOOST COMMENTA QUESTE RIGA
    df_tree_test_enc = df_tree_test.copy() if df_tree_test is not None else None # SE VUOI TESTARE ALBERI DECISIONALI CHE NON SIANO CAT BOOST O XGBOOST COMMENTA QUESTE RIGA
    return EncodingResult(
        df_tree_train_enc, 
        df_nn_train_enc, 
        df_tree_test_enc, 
        df_nn_test_enc)