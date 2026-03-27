import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


#  Caricamento dati
def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


#  Preparazione dati + scaling
def prepare_data(train_df):
    X = train_df.drop(["Transported", "PassengerId"], axis=1)
    y = train_df["Transported"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def prepare_test(test_df, scaler):
    X_test = test_df.drop("PassengerId", axis=1)
    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled


#  Creazione modello MLP
def create_nn_model():
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),  # 2 layer
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    return model


#  Training
def train_model(model, X, y):
    model.fit(X, y)
    return model


#  Predizione
def predict(model, X_test):
    return model.predict(X_test)


#  Valutazione
def evaluate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean()


#  Mostra predizioni
def show_predictions(test_df, predictions, n=10):
    results = test_df.copy()
    results["Predicted_Transported"] = predictions

    print(f"\n🔮 Prime {n} predizioni:\n")
    print(results[["PassengerId", "Predicted_Transported"]].head(n))

    return results


#  Submission
def create_submission(test_df, predictions, output_path="submission_nn.csv"):
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Transported": predictions
    })
    submission.to_csv(output_path, index=False)