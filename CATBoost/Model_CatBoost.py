import pandas as pd
from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# =========================
# DATA LOADING
# =========================

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def prepare_data(train_df):
    X = train_df.drop("Transported", axis=1)
    y = train_df["Transported"]
    return X, y


def prepare_test(test_df):
    return test_df.copy()


# =========================
# MODEL
# =========================

def create_catboost_model():
    model = CatBoostClassifier(
        iterations=5000,
        learning_rate=0.02,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        bootstrap_type='Bernoulli',
        subsample=0.8,
        random_seed=42,
        verbose=200,
        early_stopping_rounds=200
    )
    return model


# =========================
# TRAINING
# =========================

def train_model(model, X_train, y_train, X_val=None, y_val=None):

    cat_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            cat_features=cat_features,
            eval_set=(X_val, y_val)
        )
    else:
        model.fit(
            X_train, y_train,
            cat_features=cat_features
        )

    return model


# =========================
# PREDICTION
# =========================

def predict(model, X_test):
    probs = model.predict_proba(X_test)[:, 1]
    return (probs > 0.5).astype(int)


# =========================
# EVALUATION
# =========================

def evaluate_model(model, X_val, y_val):
    y_pred = predict(model, X_val)

    acc = accuracy_score(y_val, y_pred)

    print("\n📊 Validation results")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_val, y_pred))
    print("\nConfusion matrix:\n")
    print(confusion_matrix(y_val, y_pred))

    return acc




