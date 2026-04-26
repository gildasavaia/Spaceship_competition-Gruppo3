import pandas as pd
# 1. MODIFICA: Importiamo KFold invece di StratifiedKFold
from sklearn.model_selection import KFold 

def esegui_split_kfold_standard(df, target_col='Transported', random_state=42):
    """
    Divide il dataset utilizzando la K-Fold Cross Validation (Non Stratificata).
    """
    

    i=0
    
    while i==0:
        n_splits = input("Inserisci il numero di fold per la K-Fold Cross Validation: ").strip()
        print(f"Inizio K-Fold Standard (Folds: {n_splits}) sul dataset ")
        try:
            n_splits = int(n_splits)
            if n_splits > 1:
                i=1
            else:
                print("Errore: n_splits deve essere un numero intero maggiore di 1.")
        except ValueError:
            print("Errore: n_splits deve essere un numero intero.") 
            
        except ValueError:
            print("Errore: n_splits deve essere un numero.")
    # ==========================================
    # SEPARAZIONE FEATURES E TARGET
    # ==========================================
    X = df.drop(target_col, axis=1) # Features
    y = df[target_col]              # Target

    # ==========================================
    # INIZIALIZZAZIONE K-FOLD
    # ==========================================
    # 2. MODIFICA: Usiamo KFold. 
    # shuffle=True mescola l'intero dataset a caso prima di tagliarlo a fette.
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Lista che conterrà tutti i nostri dataset divisi
    kfold_datasets = []

    # ==========================================
    # GENERAZIONE DEGLI SPLIT
    # ==========================================
    fold_numero = 1
    
    # 3. MODIFICA: kf.split() ha bisogno solo di X, non guarda più y
    for train_index, test_index in kf.split(X):
        # Usiamo .iloc per selezionare le righe in base agli indici generati dal K-Fold
        X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        y_train, y_test = y.iloc[train_index].copy(), y.iloc[test_index].copy()
        
        # Salviamo questo specifico split nella nostra lista
        kfold_datasets.append((X_train, X_test, y_train, y_test))
        
        print(f"Fold {fold_numero} completato -> Train: {X_train.shape[0]} righe | Test: {X_test.shape[0]} righe")
        fold_numero += 1

    print("="*40 + "\n")
    
    return kfold_datasets