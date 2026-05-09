import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def pulisci_binarie(serie):
    """Converte le variabili binarie (testo o booleani) in 0 e 1."""
    return serie.map({'False': 0, 'True': 1, False: 0, True: 1}).fillna(0).astype(int)

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


#motivo per cui abbiamo eliminato deck_A
    """
Ecco perché eliminiamo Deck_A ma teniamo la diagonale a 1:

1. La diagonale a 1: Correlazione di una variabile con se stessa
I valori 1.0 sulla diagonale indicano la correlazione tra una variabile e se stessa (ad esempio, l'età rispetto all'età, o la spesa in RoomService rispetto alla spesa in RoomService).

Perché la teniamo? È semplicemente il modo in cui è strutturata matematicamente una matrice. La formula della correlazione confronta due colonne: se le due colonne sono identiche, il risultato è 1. Non crea nessun problema agli algoritmi di Machine Learning, è solo un punto di riferimento visivo.

2. Eliminare Deck_A: Correlazione perfetta tra variabili diverse (Multicollinearità)
Il problema con Deck_A non è che è correlato con se stesso, ma che è perfettamente prevedibile guardando le altre variabili.
Essendo il passeggero assegnato a un solo ponte alla volta, se tenessimo tutte le colonne (Deck_A, Deck_B, Deck_C, ecc.), si verificherebbe questa equazione matematica infallibile:

Deck_A = 1 - (Deck_B + Deck_C + Deck_D + Deck_E + Deck_F + Deck_G + Deck_T)

Se tutte le altre colonne del ponte sono 0, Deck_A deve essere per forza 1.

Perché la togliamo? Questo fenomeno si chiama Trappola delle Variabili Dummy (Dummy Variable Trap) o Multicollinearità perfetta. Se passi queste colonne a un algoritmo come la Regressione Logistica o una Rete Neurale, il modello "va in tilt" (matematicamente la matrice non è invertibile). L'algoritmo non sa a quale colonna dare il peso (importanza), perché l'informazione in Deck_A è letteralmente un "doppione" esatto dell'informazione già contenuta nell'insieme degli altri ponti.

In sintesi:
La diagonale della matrice dice: "L'Età è identica all'Età". (Informazione ovvia, ma innocua).

Tenere tutti i ponti incluso Deck_A direbbe all'algoritmo: "Guarda questa nuova feature Deck_A... che però è l'esatto opposto matematico della somma di tutte le altre". (Informazione ridondante, che blocca i calcoli matriciali di molti algoritmi).

Ecco perché drop_first=True è considerato uno standard quando si preparano i dati per il Machine Learning!


    """