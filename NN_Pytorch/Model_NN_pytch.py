import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Configurazione statica del motore computazionale su CPU
device = torch.device("cpu")


def load_data(train_path, test_path):
    """
    Esegue il caricamento da disco dei dataset di addestramento e test.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def prepare_data(train_df):
    """
    Separa il DataFrame di train estraendo la variabile di target 'Transported'
    e forzandone il tipo in valore intero.
    """
    X = train_df.drop("Transported", axis=1)
    y = train_df["Transported"].astype(int)
    return X, y


def prepare_test(test_df):
    """
    Prepara il set di test rimuovendo la colonna target se accidentalmente inclusa,
    o restituendone una copia pulita.
    """
    if "Transported" in test_df.columns:
        return test_df.drop("Transported", axis=1)
    return test_df.copy()


class NeuralNet(nn.Module):
    """
    Definizione dell'architettura Multilayer Perceptron (MLP) profonda.
    Include livelli di Batch Normalization per stabilizzare i pesi e Dropout
    per minimizzare il rischio di overfitting durante le epoche.
    """

    def __init__(self, input_dim):
        super().__init__()
        # Livello di ingresso e prima trasformazione lineare
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)  # Esclusione casuale del 30% dei nodi ad ogni passo

        # Secondo blocco di elaborazione nascosto
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)  # Esclusione casuale del 20% dei nodi ad ogni passo

        # Terzo blocco di riduzione dimensionale
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu3 = nn.ReLU()

        # Livello finale di output con attivazione Sigmoide per classificazione probabilistica binaria
        self.out = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Esegue il passaggio in avanti (forward pass) dei dati attraverso i livelli della rete.
        """
        x = self.drop1(self.relu1(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu2(self.bn2(self.fc2(x))))
        x = self.relu3(self.bn3(self.fc3(x)))
        return self.sigmoid(self.out(x))


def train_model(model, X, y, epochs=60, batch_size=64, lr=0.005):
    """
    Inizializza la pipeline di addestramento: esegue la standardizzazione Z-score,
    istanzia l'ottimizzatore AdamW, definisce i mini-batch e cicla sulle epoche.
    """
    # Inizializzazione e calcolo di media e varianza sul set di addestramento
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Conversione formale delle strutture in tensori PyTorch compatibili
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values if hasattr(y, "values") else y, dtype=torch.float32).view(-1, 1)

    # Configurazione del meccanismo di caricamento dati a blocchi (Mini-batch)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.to(device)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss per obiettivi binari

    # AdamW include una correzione nativa sul weight decay (regolarizzazione L2 sui pesi)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Algoritmo di pianificazione del tasso di apprendimento basato sulla stasi della loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    loss_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Iterazione interna sui singoli mini-batch del blocco loader
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Reset obbligatorio dei gradienti accumulati nel passo precedente
            optimizer.zero_grad()

            # Calcolo della predizione e della relativa funzione di errore
            out = model(X_batch)
            loss = criterion(out, y_batch)

            # Fase di backpropagation e aggiornamento effettivo dei pesi sinaptici
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Calcolo del valore di errore medio dell'epoca corrente
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)

        # Trasmette la loss allo scheduler per valutare un eventuale decadimento del learning rate
        scheduler.step(avg_loss)

        # Stampa informativa a intervalli regolari di 10 epoche
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Memorizzazione degli oggetti all'interno dell'istanza del modello per utilizzi futuri
    model.loss_history = loss_history
    model.scaler = scaler

    return model


def predict(model, X_test):
    """
    Applica lo scaler memorizzato, esegue il pass in avanti senza calcolo del gradiente
    e restituisce un vettore monodimensionale flat contenente le classi predette (0 o 1).
    """
    model.eval()

    # Trasformazione dei dati basata esclusivamente su media/varianza calcolate nel train
    X_test = model.scaler.transform(X_test)
    X_tensor = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X_tensor)
        # Binarizzazione puntuale basata sul superamento del valore di soglia 0.5
        preds = (outputs > 0.5).int().numpy()

    # Appiattisce la matrice di output trasformandola in un array monodimensionale
    return preds.flatten()