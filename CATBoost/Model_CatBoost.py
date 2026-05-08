import pandas as pd
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score



def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test



def prepare_data(train_df):
    X = train_df.drop("Transported", axis=1)
    y = train_df["Transported"]
    return X, y


def prepare_test(test_df):
    X_test = test_df.copy()
    return X_test



def create_catboost_model():
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        eval_metric="Logloss",
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50
    )
    return model



def train_model(model, X_train, y_train, X_val=None, y_val=None):
    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
    else:
        model.fit(X_train, y_train)
    return model



def predict(model, X_test):
    return model.predict(X_test)



def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print(f"\n📊 Validation accuracy: {acc:.4f}")
    return acc




def show_predictions(test_df, predictions, n=10, show_counts=True):
    results = test_df.copy()
    results["Predicted_Transported"] = predictions

    print(f"\n🔮 Prime {n} predizioni:\n")
    print(results.head(n))

    if show_counts:
        print("\n📊 Distribuzione predizioni:")
        print(results["Predicted_Transported"].value_counts())

    return results


