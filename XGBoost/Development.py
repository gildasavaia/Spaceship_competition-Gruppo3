import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

#  Caricamento dati
def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


#  Preparazione feature e target
def prepare_data(train_df):
    X = train_df.drop(["Transported", "PassengerId"], axis=1)
    y = train_df["Transported"]
    return X, y


def show_predictions(test_df, predictions, n=10):
    """
    Mostra le prime n predizioni insieme ai PassengerId

    Parameters:
    - test_df: dataframe originale del test (con PassengerId)
    - predictions: array delle predizioni
    - n: numero di righe da mostrare
    """

    results = test_df.copy()
    results["Predicted_Transported"] = predictions

    print(f"\n🔮 Prime {n} predizioni:\n")
    print(results[["PassengerId", "Predicted_Transported"]].head(n))

    return results
#  Preparazione test set
def prepare_test(test_df):
    X_test = test_df.drop("PassengerId", axis=1)
    return X_test


#  Creazione modello base
def create_model():
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    return model


#  Creazione modello ottimizzato
def create_optimized_model():
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_lambda=1,
        reg_alpha=0.5,
        random_state=42,
        eval_metric='logloss'

    )
    return model


#  Training
def train_model(model, X, y):
    model.fit(X, y)
    return model


#  Predizione
def predict(model, X_test):
    return model.predict(X_test)


#  Valutazione con cross-validation
def evaluate_model(model, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)

        scores.append(acc)

    print("Accuracy per fold:", scores)
    print("Mean accuracy:", np.mean(scores))
    print("Std:", np.std(scores))

    return np.mean(scores)




# 🔹 Creazione submission
def create_submission(test_df, predictions, output_path="../outputs/submission_XGBoost.csv"):
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Transported": predictions
    })

    submission.to_csv(output_path, index=False)


