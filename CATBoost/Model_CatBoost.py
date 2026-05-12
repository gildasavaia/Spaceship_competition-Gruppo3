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
    return CatBoostClassifier(
        # Core
        iterations=2000,
        learning_rate=0.03,
        depth=6,

        # Quality
        loss_function='Logloss',
        eval_metric='Accuracy',

        # Speed
        thread_count=-1,
        bootstrap_type='Bernoulli',
        subsample=0.8,

        # Regularization
        l2_leaf_reg=5,

        # Overfitting control
        random_strength=1,
        od_type='Iter',
        od_wait=100,

        # Misc
        verbose=100
    )



def train_model(model, X_train, y_train):

    cat_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    model.fit(
        X_train, y_train,
        cat_features=cat_features,
        verbose=200
    )

    return model


# =========================
# PREDICTION
# =========================

def predict(model, X):
    probs = model.predict_proba(X)[:, 1]
    return (probs > 0.5).astype(int)


# =========================
# EVALUATION (HOLDOUT TEST)
# =========================

def evaluate_model(model, X_test, y_test):
    y_pred = predict(model, X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\n Test results")
    print(f"Accuracy: {acc:.4f}")

    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred))

    print("\nConfusion matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    return accuracy_score