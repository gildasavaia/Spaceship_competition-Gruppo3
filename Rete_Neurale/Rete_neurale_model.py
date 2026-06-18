import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import seaborn as sns


# ---------------------------
# 🔹 Caricamento dati
# ---------------------------
def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test


# ---------------------------
# 🔹 Preparazione train
# ---------------------------
def prepare_data(train_df):
    X = train_df.drop("Transported", axis=1)
    y = train_df["Transported"]

    return X, y


# ---------------------------
# 🔹 Preparazione test
# ---------------------------
def prepare_test(test_df):
    if "Transported" in test_df.columns:
        X_test = test_df.drop("Transported", axis=1)
    else:
        X_test = test_df.copy()

    return X_test


# ---------------------------
# 🔹 Creazione rete neurale
# ---------------------------
def create_pipeline_model():
    model = Pipeline([

        ('scaler', StandardScaler()),

        ('mlp', MLPClassifier(

            hidden_layer_sizes=(128, 64),

            activation='relu',

            solver='adam',

            alpha=0.0005,

            learning_rate='adaptive',

            max_iter=2000,

            early_stopping=True,

            validation_fraction=0.1,

            n_iter_no_change=50,

            random_state=42,
            verbose=True
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


