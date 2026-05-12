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

        iterations=1400,
        learning_rate=0.035,

        # =================================
        # TREE COMPLEXITY
        # =================================

        depth=5,

        # =================================
        # REGULARIZATION
        # =================================

        l2_leaf_reg=7,

        # =================================
        # RANDOMIZATION
        # =================================

        random_strength=0.8,

        # =================================
        # SAMPLING
        # =================================

        bootstrap_type='Bernoulli',
        subsample=0.8,

        # =================================
        # FEATURE RANDOMNESS
        # =================================

        rsm=0.85,

        # =================================
        # LOSS / METRICS
        # =================================

        loss_function='Logloss',
        eval_metric='Accuracy',

        # =================================
        # OVERFITTING CONTROL
        # =================================

        od_type='Iter',
        od_wait=80,

        # =================================
        # CPU OPTIMIZATION
        # =================================

        thread_count=-1,

        # =================================
        # SPEED
        # =================================

        leaf_estimation_iterations=1,

        # =================================
        # STABILITY
        # =================================

        random_seed=42,

        # =================================
        # OUTPUT
        # =================================

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

