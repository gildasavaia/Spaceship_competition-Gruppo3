import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(train_path, test_path):
    """
    Legge e carica in memoria i dataset in formato CSV.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def prepare_data(train_df):
    """
    Rimuove la variabile target 'Transported' dall'insieme delle feature
    del set di addestramento e la restituisce separatamente.
    """
    X = train_df.drop("Transported", axis=1)
    y = train_df["Transported"]
    return X, y


def prepare_test(test_df):
    """
    Isola le feature di test escludendo il target reale 'Transported' se presente,
    garantendo l'allineamento dimensionale con il set di addestramento.
    """
    if "Transported" in test_df.columns:
        X_test = test_df.drop("Transported", axis=1)
    else:
        X_test = test_df.copy()
    return X_test


def create_model():
    """
    Istanzia e configura un classificatore XGBClassifier definendo in modo rigoroso
    gli iperparametri per contrastare la varianza e l'overfitting degli alberi.
    """
    model = XGBClassifier(
        n_estimators=3500,          # Numero massimo di iterazioni di boosting (alberi sequenziali)
        learning_rate=0.02,         # Tasso di contrazione dell'aggiornamento dei pesi degli alberi
        max_depth=5,                # Massima profondità consentita per ogni singolo albero di decisione
        min_child_weight=1,         # Somma minima del peso delle istanze richiesta in un nodo figlio
        gamma=0.05,
        subsample=0.9,              # Frazione di righe campionata casualmente per ogni albero
        colsample_bytree=0.85,      # Frazione di colonne campionata casualmente per ogni albero
        reg_lambda=3,               # Termine di regolarizzazione L2 sui pesi delle foglie
        reg_alpha=0.3,              # Termine di regolarizzazione L1 sui pesi delle foglie
        tree_method="hist",         # Algoritmo basato su istogrammi per velocizzare il calcolo dei nodi
        max_bin=256,
        n_jobs=-1,                  # Impiego di tutti i thread di calcolo logici disponibili della CPU
        enable_categorical=True,    # Supporto nativo integrato di XGBoost per le colonne di tipo category
        objective='binary:logistic',# Funzione obiettivo per restituire probabilità di classificazione binaria
        eval_metric='logloss',      # Metrica per monitorare la binarizzazione dell'errore
        random_state=42             # Seed di blocco per la riproducibilità statistica dei risultati
    )
    return model


def fix_categorical_dtype(X):
    """
    Individua tutte le colonne testuali (object) all'interno del DataFrame
    e ne modifica il tipo in 'category' per soddisfare i requisiti di XGBoost.
    """
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")
    return X


def train_model(model, X, y):
    """
    Effettua l'addestramento e l'ottimizzazione degli alberi del modello XGBoost.
    """
    model.fit(X, y)
    return model


def predict(model, X_test):
    """
    Genera e restituisce il vettore delle etichette binarie predette (0 o 1).
    """
    return model.predict(X_test)


def save_submission(model, X_test, original_test_df, filename="submission_xgboost.csv"):
    """
    Applica il modello addestrato sulle feature di test, converte le classi in
    valori booleani (True/False) e scrive il file di sottomissione per Kaggle.
    """
    # Creazione di una copia di sicurezza per evitare modifiche involontarie al DataFrame originario
    X_input = fix_categorical_dtype(X_test.copy())

    # Rimozione dell'identificativo passeggero per evitare che venga usato erroneamente come feature
    if "PassengerId" in X_input.columns:
        X_input = X_input.drop("PassengerId", axis=1)

    # Generazione delle predizioni binarie discrete
    preds = predict(model, X_input)

    # Generazione della struttura tabellare formattata secondo i requisiti della competizione
    submission = pd.DataFrame({
        "PassengerId": original_test_df["PassengerId"],
        "Transported": preds.astype(bool)
    })

    # Scrittura fisica del file su disco senza esportare l'indice di riga di pandas
    submission.to_csv(filename, index=False)
    print(f"\n Submission salvata correttamente in: {filename}")