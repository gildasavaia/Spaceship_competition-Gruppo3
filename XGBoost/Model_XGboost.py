import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test



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



def create_model():
    model = XGBClassifier(

        n_estimators=3500,
        learning_rate=0.02,
        max_depth=5,
        min_child_weight=1,

        gamma=0.05,

        subsample=0.9,
        colsample_bytree=0.85,

        reg_lambda=3,
        reg_alpha=0.3,
        tree_method="hist",
        max_bin=256,
        n_jobs=-1,
        enable_categorical=True,

        objective='binary:logistic',
        eval_metric='logloss',

        random_state=42

    )
    return model

def fix_categorical_dtype(X):
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")
    return X

def train_model(model, X, y):
    model.fit(X, y)
    return model



#
def predict(model, X_test):
    return model.predict(X_test)


def save_submission(model, X_test, original_test_df, filename="submission_xgboost.csv"):
    """
    Genera il file CSV per Kaggle.
    Converte le predizioni in Booleani (True/False).
    """
    # Assicuriamoci che i tipi categorici siano corretti
    X_input = fix_categorical_dtype(X_test.copy())

    # Rimuoviamo PassengerId se presente nelle feature di input
    if "PassengerId" in X_input.columns:
        X_input = X_input.drop("PassengerId", axis=1)

    # Otteniamo le predizioni
    preds = predict(model, X_input)

    # Creiamo il DataFrame nel formato Kaggle
    submission = pd.DataFrame({
        "PassengerId": original_test_df["PassengerId"],
        "Transported": preds.astype(bool)
    })

    submission.to_csv(filename, index=False)
    print(f"\n💾 Submission XGBoost salvata con successo in: {filename}")

