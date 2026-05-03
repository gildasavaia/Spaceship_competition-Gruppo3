import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import accuracy_score
# ---------------------------
# 🔹 Caricamento dati
# ---------------------------
def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


# ---------------------------
# 🔹 Preparazione feature e target
# ---------------------------
def prepare_data(train_df):
    X = train_df.drop(["Transported", "PassengerId"], axis=1)
    y = train_df["Transported"]
    return X, y


def prepare_test(test_df):
    X_test = test_df.drop("PassengerId", axis=1)
    return X_test


# ---------------------------
# 🔹 Visualizzazione predizioni
# ---------------------------
def show_predictions(test_df, predictions, n=10, show_counts=True):
    results = test_df.copy()
    results["Predicted_Transported"] = predictions

    print(f"\n🔮 Prime {n} predizioni:\n")
    print(results[["PassengerId", "Predicted_Transported"]].head(n))

    if show_counts:
        print("\n📊 Distribuzione predizioni:")
        print(results["Predicted_Transported"].value_counts())

    return results


# ---------------------------
# 🔹 Creazione modelli
# ---------------------------
def create_catboost_model():
    """
    Crea modello CatBoost con early stopping
    """
    model = CatBoostClassifier(
        iterations=1000,         # numero massimo di alberi
        learning_rate=0.03,      # passo di apprendimento
        depth=6,                 # profondità alberi
        eval_metric='Logloss',   # metrica di loss
        random_seed=42,
        verbose=100,             # stampa ogni 100 iterazioni
        early_stopping_rounds=50 # ferma se non migliora
    )
    return model


# ---------------------------
# 🔹 Training
# ---------------------------
def train_model(model, X_train, y_train, X_val=None, y_val=None):
    """
    Se X_val e y_val sono forniti, CatBoost usa early stopping automaticamente
    """
    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
    else:
        model.fit(X_train, y_train)
    return model


# ---------------------------
# 🔹 Predizione
# ---------------------------
def predict(model, X_test):
    return model.predict(X_test)


# ---------------------------
# 🔹 Valutazione con cross-validation
# ---------------------------
def evaluate_model(model, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)

        scores.append(acc)

    print("Accuracy per fold:", scores)
    print("Mean accuracy:", np.mean(scores))

    return np.mean(scores)
# ---------------------------
# 🔹 Feature importance
# ---------------------------
def plot_feature_importance(model, feature_names):
    importances = model.get_feature_importance()
    plt.barh(feature_names, importances)
    plt.xlabel("Importanza delle feature")
    plt.title("Feature Importance - CatBoost")
    plt.show()


# ---------------------------
# 🔹 Creazione submission
# ---------------------------
def create_submission(test_df, predictions, output_path="../outputs/submission_catboost.csv"):
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Transported": predictions
    })
    submission.to_csv(output_path, index=False)


