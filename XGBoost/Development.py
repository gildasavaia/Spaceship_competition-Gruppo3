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
        n_estimators=1200,
        learning_rate=0.05,

        max_depth=7,

        min_child_weight=3,

        subsample=0.8,
        colsample_bytree=0.8,

        gamma=0.3,

        reg_lambda=1,
        reg_alpha=0.2,

        random_state=42,

        eval_metric='logloss'
    )
    return model



def train_model(model, X, y):
    model.fit(X, y)
    return model


#
def predict(model, X_test):
    return model.predict(X_test)



def evaluate_on_test(model, X_test, y_test):
    """
    Usa il test (che nel tuo caso ha la label)
    """
    from sklearn.metrics import accuracy_score

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n Accuracy sul test: {acc:.4f}")
    return acc



def show_predictions(predictions, n=10):
    print(f"\n🔮 Prime {n} predizioni:\n")
    print(predictions[:n])


def show_confusion_matrix(model, X_test, y_test):
    """
    Mostra confusion matrix e classification report

    Args:
        model: modello addestrato
        X_test: feature del test
        y_test: target reale
    """

    # Predizioni
    y_pred = model.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # Grafico confusion matrix
    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=["False", "True"],
        yticklabels=["False", "True"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.show()
