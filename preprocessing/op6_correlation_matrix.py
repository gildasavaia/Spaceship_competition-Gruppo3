import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def pulisci_binarie(serie):
    """Converte le variabili binarie (testo o booleani) in 0 e 1."""
    return serie.replace({'False': 0, 'True': 1, False: 0, True: 1}).fillna(0).astype(int)

def run_op6(df, config=None):
    """
    Calcola la matrice di correlazione estesa (tutte le feature visibili per l'analisi).
    """
    if config is None:
        config = {"output_dir": "."}
    
    # Creiamo la cartella di output se non esiste
    output_dir = config.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)
    
    # Lavoriamo su una copia per non modificare il df originale
    df_clean = df.copy()

    # --- 1. PULIZIA E SPLIT DI CABIN ---
    if 'Name' in df_clean.columns:
        df_clean = df_clean.drop(columns=['Name']) # Rimuoviamo Name

    if 'Cabin' in df_clean.columns:
        # Dividiamo Cabin in 3 colonne
        df_clean[['Deck', 'Num', 'Side']] = df_clean['Cabin'].str.split('/', expand=True)
        df_clean['Num'] = pd.to_numeric(df_clean['Num'], errors='coerce')
        df_clean = df_clean.drop(columns=['Cabin']) # Cabin originale non serve più

    # --- 2. GESTIONE VARIABILI BINARIE ORIGINALI ---
    for col in ['CryoSleep', 'VIP', 'Transported']:
        if col in df_clean.columns:
            df_clean[col] = pulisci_binarie(df_clean[col])

    # --- 3. ENCODING COMPLETO (Senza Drop_first per esplorare tutto) ---
    categorical_cols = ['HomePlanet', 'Destination', 'Deck', 'Side']
    cols_to_encode = [col for col in categorical_cols if col in df_clean.columns]
    
    if cols_to_encode:
        # Nota: drop_first=False così vediamo P/S, Earth/Mars/Europa, ecc.
        df_encoded = pd.get_dummies(df_clean, columns=cols_to_encode, drop_first=False, dtype=int)
    else:
        df_encoded = df_clean

    # Assicuriamoci di calcolare la correlazione solo sulle colonne numeriche
    df_numerico = df_encoded.select_dtypes(include=['number', 'bool'])

    # --- 4. CALCOLO DELLA MATRICE DI CORRELAZIONE ---
    corr_matrix = df_numerico.corr()

    # --- 5. VISUALIZZAZIONE E SALVATAGGIO ---
    # Aumentiamo leggermente la grandezza della figura perché ora ci sono più colonne
    plt.figure(figsize=(24, 20)) 
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Matrice di Correlazione (Esplorazione Completa)', fontsize=24)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    # Salva il grafico nella cartella specificata
    save_path = os.path.join(output_dir, 'correlation_matrix.png')
    plt.savefig(save_path, dpi=300)
    plt.close() # Chiude la figura per liberare memoria
    
    # --- 6. ESTRAZIONE CORRELAZIONE RISPETTO AL TARGET ---
    corr_target = None
    if 'Transported' in corr_matrix.columns:
        # Prende le correlazioni con 'Transported', in valore assoluto, in ordine decrescente
        corr_target = corr_matrix['Transported'].abs().sort_values(ascending=False)
        
    return {
        "corr_matrix": corr_matrix,
        "corr_with_target": corr_target
    }
