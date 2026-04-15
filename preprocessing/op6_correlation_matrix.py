import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def pulisci_binarie(serie):
    """Converte le variabili binarie (testo o booleani) in 0 e 1."""
    return serie.replace({'False': 0, 'True': 1, False: 0, True: 1}).fillna(0).astype(int)

def run_op6(df, config=None):
    """
    Calcola la matrice di correlazione con One-Hot Encoding per le categoriche.
    """
    if config is None:
        config = {"output_dir": "."}
    
    # Creiamo la cartella di output se non esiste
    output_dir = config.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)
    
    # Lavoriamo su una copia per non modificare il df originale
    df_encoded = df.copy()

    # 1. Gestione Variabili Binarie
    for col in ['CryoSleep', 'VIP', 'Transported']:
        if col in df_encoded.columns:
            df_encoded[col] = pulisci_binarie(df_encoded[col])

    if 'Side' in df_encoded.columns:
        df_encoded['Side'] = df_encoded['Side'].map({'P': 0, 'S': 1}).fillna(0).astype(int)

    # 2. Gestione Variabili Categoriche (One-Hot Encoding)
    categorical_cols = ['HomePlanet', 'Destination', 'Deck']
    cols_to_encode = [col for col in categorical_cols if col in df_encoded.columns]
    
    if cols_to_encode:
        df_encoded = pd.get_dummies(df_encoded, columns=cols_to_encode, drop_first=True)

    # Assicuriamoci di calcolare la correlazione solo sulle colonne numeriche
    df_numerico = df_encoded.select_dtypes(include=['number', 'bool'])

    # 3. Calcolo della Matrice di Correlazione
    corr_matrix = df_numerico.corr()

    # 4. Visualizzazione e Salvataggio
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Matrice di Correlazione', fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    # Salva il grafico nella cartella specificata dalla configurazione
    save_path = os.path.join(output_dir, 'correlation_matrix.png')
    plt.savefig(save_path, dpi=300)
    plt.close() # Chiude la figura per liberare memoria
    
    # 5. Estrazione della correlazione rispetto al target
    corr_target = None
    if 'Transported' in corr_matrix.columns:
        # Prende le correlazioni con 'Transported', in valore assoluto, le ordina in modo decrescente
        corr_target = corr_matrix['Transported'].abs().sort_values(ascending=False)
        
    # Restituiamo il dizionario atteso dalla tua "nuova pipeline"
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