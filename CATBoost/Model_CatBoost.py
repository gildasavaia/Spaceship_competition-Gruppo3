import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import accuracy_score



def load_data(train_path, test_path):
    """
    Carica i dataset di training e test da file CSV

    Args:
        train_path: percorso file train
        test_path: percorso file test

    Returns:
        train, test: DataFrame pandas
    """
    train = pd.read_csv(train_path)  # legge il dataset di training
    test = pd.read_csv(test_path)    # legge il dataset di test
    return train, test


#
def prepare_data(train_df):
    """
    Separa features (X) e target (y)

    - Rimuove colonne inutili
    - Target = "Transported"

    Args:
        train_df: DataFrame di training

    Returns:
        X: feature
        y: target
    """
    X = train_df.drop(["Transported", "PassengerId"], axis=1)  # rimuove target e ID
    y = train_df["Transported"]  # target da predire
    return X, y



def prepare_test(test_df):
    """
    Prepara il dataset di test rimuovendo colonne non utili

    Args:
        test_df: DataFrame test

    Returns:
        X_test: feature del test
    """
    X_test = test_df.drop("PassengerId", axis=1)  # rimuove ID (non serve al modello)
    return X_test



def show_predictions(test_df, predictions, n=10, show_counts=True):
    """
    Mostra un'anteprima delle predizioni

    Args:
        test_df: dataset originale
        predictions: output del modello
        n: numero di righe da mostrare
        show_counts: mostra distribuzione classi

    Returns:
        DataFrame con predizioni
    """
    results = test_df.copy()  # copia per non modificare originale
    results["Predicted_Transported"] = predictions  # aggiunge colonna predetta

    # stampa prime n predizioni
    print(f"\n🔮 Prime {n} predizioni:\n")
    print(results[["PassengerId", "Predicted_Transported"]].head(n))

    # stampa distribuzione classi (utile per capire bias)
    if show_counts:
        print("\n📊 Distribuzione predizioni:")
        print(results["Predicted_Transported"].value_counts())

    return results



def create_catboost_model():
    """
    Crea un modello CatBoost configurato

    Include:
    - early stopping
    - parametri base ottimizzati
    """
    model = CatBoostClassifier(
        iterations=1000,         # numero massimo di alberi
        learning_rate=0.03,      # quanto velocemente apprende
        depth=6,                 # complessità alberi
        eval_metric='Logloss',   # funzione di loss
        random_seed=42,          # per risultati riproducibili
        verbose=100,             # stampa progresso ogni 100 iterazioni
        early_stopping_rounds=50 # stop se non migliora per 50 iterazioni
    )
    return model


def train_model(model, X_train, y_train, X_val=None, y_val=None):
    """
    Addestra il modello

    Se viene fornito validation set:
    → usa early stopping automaticamente

    Args:
        model: modello CatBoost
        X_train, y_train: dati training
        X_val, y_val: dati validation (opzionale)

    Returns:
        modello addestrato
    """
    if X_val is not None and y_val is not None:
        # training con validation (early stopping attivo)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
    else:
        # training senza validation
        model.fit(X_train, y_train)

    return model


def predict(model, X_test):
    """
    Genera predizioni sul test set

    Args:
        model: modello addestrato
        X_test: dati test

    Returns:
        predizioni
    """
    return model.predict(X_test)



def evaluate_model(model, X, y, cv=5):
    """
    Valuta il modello con K-Fold Cross Validation

    Processo:
    - divide i dati in k fold
    - allena su k-1
    - valida su 1
    - ripete

    Args:
        model: modello
        X, y: dataset completo
        cv: numero di fold

    Returns:
        accuracy media
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X):
        # split manuale
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # training con validation
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

        # predizione
        y_pred = model.predict(X_val)

        # calcolo accuracy
        acc = accuracy_score(y_val, y_pred)
        scores.append(acc)

    print("Accuracy per fold:", scores)
    print("Mean accuracy:", np.mean(scores))

    return np.mean(scores)



def plot_feature_importance(model, feature_names):
    """
    Mostra importanza delle feature

    Args:
        model: modello addestrato
        feature_names: nomi colonne
    """
    importances = model.get_feature_importance()  # ottiene importanza

    # grafico orizzontale
    plt.barh(feature_names, importances)
    plt.xlabel("Importanza delle feature")
    plt.title("Feature Importance - CatBoost")
    plt.show()


def create_submission(test_df, predictions, output_path="../outputs/submission_catboost.csv"):
    """
    Crea file CSV per submission (es. Kaggle)

    Args:
        test_df: dataset test originale
        predictions: predizioni modello
        output_path: dove salvare il file
    """
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],  # ID richiesto
        "Transported": predictions              # target predetto
    })

    submission.to_csv(output_path, index=False)  # salva CSV