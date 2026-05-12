import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler


# =========================================================
# 🔹 DEVICE (CPU ONLY)
# =========================================================

device = torch.device("cpu")


# =========================================================
# 🔹 LOAD DATA
# =========================================================

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


# =========================================================
# 🔹 PREPARE DATA
# =========================================================

def prepare_data(train_df):
    X = train_df.drop("Transported", axis=1)
    y = train_df["Transported"].astype(int)
    return X, y


def prepare_test(test_df):
    if "Transported" in test_df.columns:
        return test_df.drop("Transported", axis=1)
    return test_df.copy()


# =========================================================
# 🔹 MODELLO NEURALE
# =========================================================

class NeuralNet(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# 🔹 TRAINING
# =========================================================

def train_model(model, X, y, epochs=50, lr=0.001, batch_size=64):

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    for epoch in range(epochs):

        total_loss = 0

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            out = model(X_batch)
            loss = criterion(out, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    model.loss_history = loss_history
    model.scaler = scaler

    return model


# =========================================================
# 🔹 PREDICT
# =========================================================

def predict(model, X_test):

    model.eval()

    X_test = model.scaler.transform(X_test)

    X_tensor = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X_tensor)
        preds = (outputs > 0.5).int().numpy()

    return preds.flatten()