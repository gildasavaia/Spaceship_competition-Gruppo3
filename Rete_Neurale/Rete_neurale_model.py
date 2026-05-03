import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score



def load_data(train_path, test_path):
    """
       Carica i dataset di training e test da file CSV.

       Parameters:
       - train_path: percorso del file di training
       - test_path: percorso del file di test

       Returns:
       - train: DataFrame del training set
       - test: DataFrame del test set
       """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test



def prepare_data(train_df):
    """
       Prepara i dati di training separando feature e target.

       Rimuove:
       - PassengerId (non informativo per il modello)
       - Transported (target)

       Parameters:
       - train_df: DataFrame completo di training

       Returns:
       - X: features (input del modello)
       - y: target (variabile da predire)
       """
    X = train_df.drop(["Transported", "PassengerId"], axis=1)
    y = train_df["Transported"]
    return X, y


def prepare_test(test_df):
    """
       Prepara il dataset di test rimuovendo le colonne non utili al modello.

       Rimuove:
       - PassengerId

       Parameters:
       - test_df: DataFrame del test set

       Returns:
       - X_test: feature del test set
       """
    X_test = test_df.drop("PassengerId", axis=1)
    return X_test



def create_pipeline_model():
    """
       Crea una pipeline completa che include:
       - StandardScaler: normalizza i dati (fondamentale per MLP)
       - MLPClassifier: rete neurale feed-forward

       Architettura rete:
       - 2 layer nascosti: 64 neuroni e 32 neuroni
       - ReLU come funzione di attivazione
       - Ottimizzatore Adam

       Altri parametri:
       - max_iter=500: massimo numero di epoche
       - early_stopping=True: ferma training se non migliora
       - n_iter_no_change=20: pazienza per early stopping
       - random_state=42: riproducibilità

       Returns:
       - Pipeline pronta per training e prediction
       """
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42
        ))
    ])
    return model



def train_model(model, X, y):
    """
      Allena il modello sui dati di training.

      Parameters:
      - model: pipeline (scaler + rete neurale)
      - X: features
      - y: target

      Returns:
      - model allenato
      """
    model.fit(X, y)
    return model



def predict(model, X_test):
    """
       Genera predizioni sul test set.

       Parameters:
       - model: modello già allenato
       - X_test: dati di test

       Returns:
       - array di predizioni (0/1 o False/True)
       """
    return model.predict(X_test)


def evaluate_model(model, X, y, cv=5):
    """
       Valuta il modello usando Stratified K-Fold Cross Validation.

       StratifiedKFold:
       - mantiene la proporzione delle classi in ogni fold

       Parameters:
       - model: pipeline da valutare
       - X: features
       - y: target
       - cv: numero di fold (default 5)

       Returns:
       - media accuracy sui fold
       """

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

    print("\n Risultati Cross-Validation:")
    print("Accuracy per fold:", scores)
    print("Mean accuracy:", scores.mean())
    print("Std:", scores.std())

    return scores.mean()



def show_predictions(test_df, predictions, n=10):
    """
        Mostra un esempio delle predizioni del modello.

        Parameters:
        - test_df: dataset originale test
        - predictions: output del modello
        - n: numero di righe da mostrare

        Returns:
        - DataFrame con le predizioni aggiunte
        """
    results = test_df.copy()
    results["Predicted_Transported"] = predictions

    print(f"\n🔮 Prime {n} predizioni:\n")
    print(results[["PassengerId", "Predicted_Transported"]].head(n))

    return results



def create_submission(test_df, predictions, output_path="../datas/submission_rete_neurale.csv"):
    """
        Crea il file di submission (tipico per competizioni tipo Kaggle).

        Formato richiesto:
        - PassengerId
        - Transported (predizione)

        Parameters:
        - test_df: dataset originale test
        - predictions: output del modello
        - output_path: percorso dove salvare il CSV

        Returns:
        - salva un file CSV pronto per submission
        """

    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Transported": predictions
    })

    submission.to_csv(output_path, index=False)
    print(f" File salvato in: '{output_path}'")