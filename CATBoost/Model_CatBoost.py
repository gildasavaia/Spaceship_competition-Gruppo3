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
        depth=5,
        l2_leaf_reg=7,
        random_strength=0.8,
        bootstrap_type='Bernoulli',
        subsample=0.8,
        rsm=0.85,

        loss_function='Logloss',
        eval_metric='Accuracy',
        od_type='Iter',
        od_wait=80,
        thread_count=-1,
        leaf_estimation_iterations=1,
        random_seed=42,
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
    # Identifichiamo le colonne categoriche come fatto nel training
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    # Passiamo esplicitamente le cat_features anche in fase di predizione
    # Se il modello è già stato addestrato con cat_features,
    # a volte basta passarle correttamente tramite un oggetto Pool
    from catboost import Pool
    test_pool = Pool(data=X, cat_features=cat_features)

    probs = model.predict_proba(test_pool)[:, 1]
    return (probs > 0.5).astype(int)


def save_submission(model, X_test, original_test_df, filename="submission_catboost.csv"):
    # Rimuoviamo PassengerId dai dati di input se presente,
    # perché il modello non è stato addestrato su di esso
    X_input = X_test.drop("PassengerId", axis=1) if "PassengerId" in X_test.columns else X_test

    preds = predict(model, X_input)

    submission = pd.DataFrame({
        "PassengerId": original_test_df["PassengerId"],
        "Transported": preds.astype(bool)
    })

    submission.to_csv(filename, index=False)
    print(f"\n💾 Submission salvata in: {filename}")