import pandas as pd
import itertools
from pathlib import Path

def prepara_dati(df):
    """Estrae le feature secondarie come Ponte, Lato, Gruppo e Cognome."""
    # Split Cabina
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    # Split PassengerId
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    # Split Nome
    df[['Nome_Proprio', 'Surname']] = df['Name'].str.split(' ', expand=True)
    return df

def trova_regole_esclusione(df, feature_categoriche):
    """
    Cerca le combinazioni impossibili (frequenza = 0) tra tutte le coppie
    di variabili categoriche.
    """
    print("="*60)
    print("🚀 1. REGOLE DI ESCLUSIONE (Cosa NON succede MAI)")
    print("="*60)
    
    # Genera tutte le possibili coppie di colonne
    coppie = list(itertools.combinations(feature_categoriche, 2))
    
    regole_trovate = 0
    for col1, col2 in coppie:
        # Usiamo crosstab per creare una matrice delle frequenze
        tabella = pd.crosstab(df[col1], df[col2])
        
        # Iteriamo sulle righe e colonne della matrice
        for val1 in tabella.index:
            for val2 in tabella.columns:
                if tabella.loc[val1, val2] == 0:
                    print(f"🔹 {col1}='{val1}' e {col2}='{val2}' -> MAI INSIEME (0 casi)")
                    regole_trovate += 1
                    
    if regole_trovate == 0:
        print("Nessuna regola di esclusione trovata per queste feature.")
    else:
        print(f"\n-> Trovate {regole_trovate} regole assolute di esclusione.")

def analizza_regole_gruppi_famiglie(df):
    """
    Verifica quanto sono forti le regole di coesione per chi viaggia insieme
    (Stesso Gruppo o Stesso Cognome).
    """
    print("\n" + "="*60)
    print("👨‍👩‍👧‍👦 2. REGOLE DI GRUPPO E FAMIGLIA (Coesione)")
    print("="*60)

    def valuta_coesione(colonna_raggruppamento, colonna_target):
        # Gruppi con più di 1 persona (no single) e senza NaN nel target
        df_validi = df.dropna(subset=[colonna_raggruppamento, colonna_target])
        dimensione_gruppi = df_validi[colonna_raggruppamento].value_counts()
        gruppi_multipli = dimensione_gruppi[dimensione_gruppi > 1].index
        
        df_multi = df_validi[df_validi[colonna_raggruppamento].isin(gruppi_multipli)]
        
        # Conta quanti target diversi ci sono in ogni gruppo
        target_unici_per_gruppo = df_multi.groupby(colonna_raggruppamento)[colonna_target].nunique()
        
        totale_gruppi = len(target_unici_per_gruppo)
        gruppi_puri = (target_unici_per_gruppo == 1).sum()
        percentuale = (gruppi_puri / totale_gruppi) * 100 if totale_gruppi > 0 else 0
        
        print(f"Persone con lo stesso '{colonna_raggruppamento}' condividono '{colonna_target}' nel {percentuale:.1f}% dei casi.")

    valuta_coesione('Group', 'HomePlanet')
    valuta_coesione('Group', 'Destination')
    valuta_coesione('Group', 'Cabin') # Per vedere se dormono nella stessa cabina
    valuta_coesione('Group', 'Side')  # Per vedere se stanno almeno dallo stesso lato
    
    print("-" * 40)
    
    # Altre relazioni con il Cognome
    valuta_coesione('Surname', 'Cabin')
    valuta_coesione('Surname', 'Deck')
    valuta_coesione('Surname', 'Side')
    valuta_coesione('Surname', 'VIP')
    valuta_coesione('Surname', 'CryoSleep')
    valuta_coesione('Surname', 'HomePlanet')
    valuta_coesione('Surname', 'Destination')

def analizza_regole_finanziarie(df):
    """
    Analizza regole logiche legate ai soldi e allo status.
    """
    print("\n" + "="*60)
    print("💰 3. REGOLE LOGICHE E FINANZIARIE")
    print("="*60)

    cost_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpent'] = df[cost_cols].sum(axis=1)

    # Regola CryoSleep -> 0 spese
    df_cryo = df[df['CryoSleep'] == True]
    spesa_media_cryo = df_cryo['TotalSpent'].mean()
    print(f"Spesa media per i passeggeri in CryoSleep: {spesa_media_cryo:.2f} crediti")
    if spesa_media_cryo > 0:
         print("   ⚠️ ATTENZIONE: Ci sono errori nel dataset, alcuni in CryoSleep hanno speso!")
    else:
         print("   ✅ CONFERMATO: Chi è in CryoSleep spende sempre 0.")

    # Età -> Spese e CryoSleep
    bambini = df[df['Age'] < 13]
    spesa_media_bambini = bambini['TotalSpent'].mean()
    print(f"Spesa media per i bambini (Età < 13): {spesa_media_bambini:.2f} crediti")

def main():
    # Carica il dataset (aggiusta il path se necessario)
    data_path = Path(__file__).resolve().parents[1] / "data" / "train.csv"
    try:
         df = pd.read_csv(data_path)
    except FileNotFoundError:
         # Fallback in caso di esecuzione dalla root
         df = pd.read_csv("data/train.csv")
         
    df = prepara_dati(df)

    # Variabili su cui cercare incroci impossibili
    feature_categoriche = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP', 'Deck', 'Side']
    
    trova_regole_esclusione(df, feature_categoriche)
    analizza_regole_gruppi_famiglie(df)
    analizza_regole_finanziarie(df)

if __name__ == "__main__":
    main()