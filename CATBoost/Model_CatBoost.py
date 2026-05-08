import pandas as pd
from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import seaborn as sns


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

        iterations=3000,

        learning_rate=0.03,

        depth=8,

        l2_leaf_reg=5,

        loss_function='Logloss',

        eval_metric='Accuracy',

        bootstrap_type='Bernoulli',

        subsample=0.8,

        random_seed=42,

        verbose=100,

        early_stopping_rounds=100
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




def show_predictions(test_df, predictions, n=10):
    """
    Mostra le prime predizioni confrontate col valore reale
    """

    results = test_df.copy()

    # aggiunge predizioni
    results["Predicted_Transported"] = predictions

    print(f"\n🔮 Prime {n} predizioni:\n")

    # se esiste il target reale
    if "Transported" in results.columns:

        print(
            results[
                ["Transported", "Predicted_Transported"]
            ].head(n)
        )

    else:
        print(
            results[
                ["Predicted_Transported"]
            ].head(n)
        )

    return results


