import pandas as pd
from sklearn.model_selection import train_test_split

def esegui_split_dati(df, target_col='Transported', random_state=42):
    """
    Divide il dataset in variabili di addestramento e test.
    Ideale per essere richiamata all'interno di una pipeline di Machine Learning.
    
    Parametri:
    - df: Il DataFrame Pandas completo (es. i dati di addestramento originali).
    - target_col: Il nome della colonna da prevedere (default: 'Transported').
    - test_size: La proporzione del dataset da dedicare al test (default: 0.2, ovvero 20%).
    - random_state: Il seed per la riproducibilità (default: 42).
    
    Ritorna:
    - X_train, X_test, y_train, y_test
    """
    i=0
    
    while i==0:
        test_size = input("Inserisci la proporzione del dataset da dedicare al test (es. 0.2 per il 20%): ").strip()
        try:
            test_size = float(test_size)
            if 0.0 < test_size < 1.0:
                i=1
            else:
                print("Errore: test_size deve essere un numero compreso tra 0 e 1.")
        except ValueError:
            print("Errore: test_size deve essere un numero.")
    # Controllo di sicurezza sulla proporzione
   
    # ==========================================
    # SEPARAZIONE FEATURES E TARGET
    # ==========================================
    X = df.drop(target_col, axis=1) # Features
    y = df[target_col]              # Target

    # ==========================================
    # SPLIT (Metodo Holdout)
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )

   
    # Ritorna le 4 variabili per i passaggi successivi della pipeline
    return X_train, X_test, y_train, y_test