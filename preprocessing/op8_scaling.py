import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class ScalingResult:
    def __init__(self, df_tree_train, df_nn_train, df_tree_test=None, df_nn_test=None, scaler=None):
        self.df_tree_train = df_tree_train
        self.df_nn_train = df_nn_train
        self.df_tree_test = df_tree_test
        self.df_nn_test = df_nn_test
        self.scaler = scaler  # Salviamo lo scaler nel caso serva in futuro

def run_scaling(df_train, df_test=None):
    """
    Applica la standardizzazione e il logaritmo.
    Restituisce due versioni dei dati: una non scalata (per Alberi) e una scalata (per Reti/Lineari).
    Evita il Data Leakage fittando lo scaler SOLO sul train.
    """
    # 1. DATASET PER ALBERI (Copia diretta, nessuna modifica)
    df_tree_train = df_train.copy()
    df_tree_test = df_test.copy() if df_test is not None else None

    # 2. DATASET PER MODELLI LINEARI E RETI NEURALI
    df_nn_train = df_train.copy()
    df_nn_test = df_test.copy() if df_test is not None else None

    # Colonne numeriche da scalare (verifica che corrispondano a quelle che hai post OP5)
    numeric_cols = ['Age', 'GroupSize', 'NumZone', 'TotalSpending', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # Filtriamo solo le colonne numeriche effettivamente presenti nel dataframe
    numeric_cols = [c for c in numeric_cols if c in df_train.columns]


    # Lo scaling per il ramo NN viene gestito internamente da train_model()
    # in Model_NN_pytch.py. Questo evita il doppio scaling e permette di
    # scalare anche le colonne one-hot in modo omogeneo.
    scaler = None

    return ScalingResult(df_tree_train, df_nn_train, df_tree_test, df_nn_test, scaler)