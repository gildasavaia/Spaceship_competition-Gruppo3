import pandas as pd
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


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
    """
    Train NON ha PassengerId
    """
    X = train_df.drop("Transported", axis=1)
    y = train_df["Transported"]
    return X, y


def prepare_test(test_df):
    """
    Nel tuo test hai già Transported (per validazione)
    → lo togliamo dalle feature
    """
    if "Transported" in test_df.columns:
        X_test = test_df.drop("Transported", axis=1)
    else:
        X_test = test_df.copy()

    return X_test


# ---------------------------
# 🔹 Modello XGBoost ottimizzato
# ---------------------------
def create_model():
    model = XGBClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_lambda=1,
        reg_alpha=0.5,
        random_state=42,
        eval_metric='logloss'
    )
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
# 🔹 Valutazione (semplice, senza CV)
# ---------------------------
def evaluate_on_test(model, X_test, y_test):
    """
    Usa il test (che nel tuo caso ha la label)
    """
    from sklearn.metrics import accuracy_score

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n📊 Accuracy sul test: {acc:.4f}")
    return acc


# ---------------------------
# 🔹 Mostra predizioni
# ---------------------------
def show_predictions(predictions, n=10):
    print(f"\n🔮 Prime {n} predizioni:\n")
    print(predictions[:n])


# ---------------------------
# 🔹 Feature importance
# ---------------------------
