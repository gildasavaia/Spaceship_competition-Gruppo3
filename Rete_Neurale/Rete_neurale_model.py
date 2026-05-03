import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score


# ---------------------------
# 🔹 Caricamento dati
# ---------------------------
def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


# ---------------------------
# 🔹 Preparazione dati
# ---------------------------
def prepare_data(train_df):
    X = train_df.drop(["Transported", "PassengerId"], axis=1)
    y = train_df["Transported"]
    return X, y


def prepare_test(test_df):
    X_test = test_df.drop("PassengerId", axis=1)
    return X_test


# ---------------------------
# 🔹 Creazione modello con pipeline
# ---------------------------
def create_pipeline_model():
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


# ---------------------------
# 🔹 Training
# ---------------------------
def train_model(model, X, y):
    model.fit(X, y)
    return model


# ---------------------------
# 🔹 Predizione
# ---------------------------
def predict(model, X_test):
    return model.predict(X_test)


# ---------------------------
# 🔹 Valutazione con Stratified CV
# ---------------------------
def evaluate_model(model, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

    print("\n📊 Risultati Cross-Validation:")
    print("Accuracy per fold:", scores)
    print("Mean accuracy:", scores.mean())
    print("Std:", scores.std())

    return scores.mean()


# ---------------------------
# 🔹 Mostra predizioni
# ---------------------------
def show_predictions(test_df, predictions, n=10):
    results = test_df.copy()
    results["Predicted_Transported"] = predictions

    print(f"\n🔮 Prime {n} predizioni:\n")
    print(results[["PassengerId", "Predicted_Transported"]].head(n))

    return results


# ---------------------------
# 🔹 Creazione submission
# ---------------------------
def create_submission(test_df, predictions, output_path="../datas/submission_rete_neurale.csv"):
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Transported": predictions
    })

    submission.to_csv(output_path, index=False)
    print(f"✅ File salvato in: '{output_path}'")