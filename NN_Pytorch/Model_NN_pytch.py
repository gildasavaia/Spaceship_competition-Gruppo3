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
        # Architettura piramidale ottimizzata con Batch Normalization e Dropout
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)  # Spegne il 30% dei neuroni per evitare overfitting

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu3 = nn.ReLU()

        self.out = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.drop1(self.relu1(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu2(self.bn2(self.fc2(x))))
        x = self.relu3(self.bn3(self.fc3(x)))
        return self.sigmoid(self.out(x))


def train_model(model, X, y, epochs=60, batch_size=64, lr=0.005):
    # Prepariamo i dati come prima
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values if hasattr(y, "values") else y, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.to(device)
    criterion = nn.BCELoss()

    # Weight decay aggiunge regolarizzazione L2 sui pesi della rete neurale
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Scheduler: riduce il lr del 50% se la loss non migliora per 5 epoche
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    loss_history = []

    for epoch in range(epochs):
        model.train()
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

        # Aggiorna lo scheduler in base alla loss media dell'epoca
        scheduler.step(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")

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