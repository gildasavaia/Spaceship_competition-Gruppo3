"""# Importa le funzioni dai file op1, op2, op3 e dal nuovo op4
from op1_read_file import run_load_and_convert_to_csv
from op2_data_evaluation import run_evaluation
from op3_split_dataset import run_split_dataset  
from op4_handler_nullvalue import run_handle_null_values # <-- NUOVO IMPORT
from pathlib import Path

# Percorso al dataset
data_path = Path(__file__).resolve().parents[1] / "data" / "train.csv"

# OP1: Lettura
risultato = run_load_and_convert_to_csv(str(data_path))
mio_dataframe = risultato.df_output

# OP3: Feature Engineering (Split)
risultato_split = run_split_dataset(mio_dataframe)
mio_dataframe_modificato = risultato_split.df_output

# OP2: Valutazione finale sui dati puliti e imputati
valutazione = run_evaluation(mio_dataframe_imputato)
print("Valutazione completata.")

# --- NUOVO STEP OP4 ---
# OP4: Gestione dei valori nulli (DEVE avvenire dopo lo split, perché usa la colonna 'Group')
risultato_imputazione = run_handle_null_values(mio_dataframe_modificato)
mio_dataframe_imputato = risultato_imputazione.df_output
# ----------------------

# Salva il DataFrame imputato
output_path = data_path.with_name("train_processed.csv")
mio_dataframe_imputato.to_csv(output_path, index=False)
print(f"Dataset finale salvato in: {output_path}")

pipeline iniziale di dario
"""


""" pipeline gilda 
from pathlib import Path
import json

from op1_read_file import run_load_and_convert_to_csv
from op2_data_evaluation import run_evaluation
from op3_split_dataset import run_split_dataset
from op4_handler_nullvalue import run_handle_null_values
from op5_sumcosts_names import run_op5
from op6_correlation_matrix import run_op6


data_path = Path(__file__).resolve().parents[1] / "data" / "train.csv"

# OP1: Lettura
risultato = run_load_and_convert_to_csv(str(data_path))
mio_dataframe = risultato.df_output

# OP3: Split / feature engineering
risultato_split = run_split_dataset(mio_dataframe)
mio_dataframe_modificato = risultato_split.df_output

# OP4: Imputazione valori nulli solo per gruppi di cardinalità 1
risultato_imputazione = run_handle_null_values(mio_dataframe_modificato)
mio_dataframe_imputato = risultato_imputazione.df_output

# OP5: Costi + rimozione nomi
risultato_op5 = run_op5(mio_dataframe_imputato)
mio_dataframe_finale = risultato_op5.df_output

TotalSpending è fondamentale → spesso è una delle feature più predittive nei dataset tipo Spaceship Titanic.
Se lasci le singole colonne + il totale → il modello può imparare meglio (non rimuoverle subito).
Se invece vuoi ridurre dimensionalità → puoi eliminare le singole colonne dopo aver creato il totale.


# --- NUOVO STEP OP6 ---
risultato_op6 = run_op6(
    mio_dataframe_finale,
    config={"output_dir": "outputs/op6"}
)

print("OP6 completato.")

if risultato_op6["corr_with_target"] is not None:
    print("\nTop correlazioni con Transported:")
    print(risultato_op6["corr_with_target"].head(10))
# ----------------------


# Salvataggio dizionari
dict_output_path = data_path.with_name("probability_dictionaries.json")
with open(dict_output_path, "w", encoding="utf-8") as f:
    json.dump(risultato_imputazione.probability_dictionaries, f, indent=4, ensure_ascii=False)

print(f"Dizionari salvati in: {dict_output_path}")

# OP2: Valutazione finale
valutazione = run_evaluation(mio_dataframe_finale)
print("Valutazione completata.")


# CORRETTO (usa mio_dataframe_finale)
output_path = data_path.with_name("train_processed.csv")
mio_dataframe_finale.to_csv(output_path, index=False)

print(f"Dataset finale salvato in: {output_path}")
"""


#nuova pipeline 
from pathlib import Path
import json
import pandas as pd

from op1_read_file import run_load_and_convert_to_csv
from op2_data_evaluation import run_evaluation
from op3_split_dataset import run_split_dataset
from op4_handler_nullvalue import run_handle_null_values
from op5_sumcosts_names import run_op5
from op6_correlation_matrix import run_op6
from op7_holdout_evaluator import esegui_split_dati as run_holdout
from op7_kfold_evaluator import esegui_split_kfold_standard as run_kfold

def main():
    print("--- INIZIO PIPELINE ---")
    
    target = "Transported"  # Colonna target per la previsione
    
    # Percorso al dataset originale
    data_path = Path(__file__).resolve().parents[1] / "data" / "train.csv"

    # OP1: Lettura
    print("\n[Esecuzione OP1] Lettura dataset...")
    risultato = run_load_and_convert_to_csv(str(data_path))
    mio_dataframe = risultato.df_output

    # OP3: Split / feature engineering (es. estrazione Deck, Num, Side da Cabin)
    print("\n[Esecuzione OP3] Split dataset e feature engineering...")
    risultato_split = run_split_dataset(mio_dataframe)
    mio_dataframe_modificato = risultato_split.df_output
    
    print("\nScegli il metodo di divisione del dataset:")
    print("1: Hold-out (Train/Test split)")
    print("2: K-Fold Cross Validation")
    scelta = input("Inserisci 1 o 2: ").strip()
    
    repository_data = data_path.parent # Punta alla cartella "preprocessed_folds" per salvare i fold elaborati
    processed_folds_path = repository_data / "preprocessed_folds"
    processed_folds_path.mkdir(parents=True, exist_ok=True)
    repository_dictionary = data_path.parent / "probability_dictionaries"
    repository_dictionary.mkdir(parents=True, exist_ok=True)

    # Ripeti la richiesta finché l'utente non inserisce una scelta valida
    while scelta not in ['1', '2', '3']:
        print("Scelta non valida. Inserisci 1 o 2.") 
        scelta = input("Inserisci 1 o 2: ").strip()
            
    
    if scelta == '1':
        print("\n[Esecuzione OP7] Divisione Hold-out...")
        # NOTA: Assumo che run_holdout restituisca due dataframe: train e test
        X_train, X_test, y_train, y_test = run_holdout(mio_dataframe_modificato)
        
        # OP4 (train): Pulizia iniziale (CryoSleep) e Imputazione valori nulli (gruppi multipli e singoli)
        print("\n[Esecuzione OP4] Gestione e imputazione valori nulli...")
        risultato_imputazione = run_handle_null_values(X_train)
        mio_dataframe_imputato = risultato_imputazione.df_output

        # Salvataggio dizionari delle probabilità (basati su OP4)
        dict_output_path = repository_dictionary / "holdout_probability_dictionaries_train.json"
        with open(dict_output_path, "w", encoding="utf-8") as f:
            json.dump(risultato_imputazione.probability_dictionaries, f, indent=4, ensure_ascii=False)
        print(f"Dizionari delle probabilità salvati in: {dict_output_path}")

        # OP5: Somma dei costi (TotalSpending) + rimozione nomi
        # Nota: TotalSpending è fondamentale e molto predittivo. 
        # Manteniamo per ora le singole colonne e creiamo il totale per ridurre/aumentare dimensionalità a scelta.
        print("\n[Esecuzione OP5] Calcolo costi totali e gestione nomi...")
        risultato_op5 = run_op5(mio_dataframe_imputato)
        X_train = risultato_op5.df_output
        
        # OP4 (test): Imputazione valori nulli sul test usando i dizionari di probabilità calcolati sul train
        print("\n[Esecuzione OP4] Gestione e imputazione valori nulli...")
        risultato_imputazione = run_handle_null_values(X_test)
        mio_dataframe_imputato = risultato_imputazione.df_output

        # Salvataggio dizionari delle probabilità (basati su OP4)
        dict_output_path = repository_dictionary / "holdout_probability_dictionaries_test.json"
        with open(dict_output_path, "w", encoding="utf-8") as f:
            json.dump(risultato_imputazione.probability_dictionaries, f, indent=4, ensure_ascii=False)
        print(f"Dizionari delle probabilità salvati in: {dict_output_path}")

        # OP5: Somma dei costi (TotalSpending) + rimozione nomi
        # Nota: TotalSpending è fondamentale e molto predittivo. 
        # Manteniamo per ora le singole colonne e creiamo il totale per ridurre/aumentare dimensionalità a scelta.
        print("\n[Esecuzione OP5] Calcolo costi totali e gestione nomi...")
        risultato_op5 = run_op5(mio_dataframe_imputato)
        X_test = risultato_op5.df_output
        
        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)
        
                # Salvataggio del dataset finale elaborato
        train_out_path = processed_folds_path / "holdout_processed_train.csv"
        test_out_path = processed_folds_path / "holdout_processed_test.csv"
        
        df_train.to_csv(train_out_path, index=False)
        df_test.to_csv(test_out_path, index=False)
        
           # OP2: Valutazione finale (sui dati completi, puliti e arricchiti)
        print("\n[Esecuzione OP2] Valutazione finale dataset...")
        valutazione = run_evaluation(df_train)  # Puoi anche valutare su df_test se vuoi vedere la performance sul test
        print("Valutazione completata.")
        
        print(f"\nn[FINE] Dataset finale salvato in: {train_out_path} e {test_out_path}")
        
        # OP6: Matrice di correlazione
        print("\n[Esecuzione OP6] Calcolo matrice di correlazione...")
        risultato_op6 = run_op6(
            mio_dataframe_modificato,
            config={"output_dir": "outputs/op6"}
        )
        print("OP6 completato.")

        # if risultato_op6["corr_with_target"] is not None:
        #     print("\nTop correlazioni con Transported:")
        #     print(risultato_op6["corr_with_target"].head(10))  
        
        return df_train, df_test
        
           
    
    elif scelta == '2':
        print("\n[OP7] Esecuzione K-Fold...")
        
        # Esegui la divisione in fold. "folds" sarà una lista contenente i dati separati per ogni iterazione.
        folds = run_kfold(mio_dataframe_modificato)
        
        # Lista per salvare le metriche (es. accuratezza) di ogni fold durante l'addestramento
        results = [] 

        # Logica di loop sui folds
        for i, fold_data in enumerate(folds):
            print(f"\n--- Elaborazione Fold {i+1} ---")
            numero_fold = i + 1
            
            # 1. Estrazione dei 4 elementi per il fold corrente
            X_train_fold, X_test_fold, y_train_fold, y_test_fold = fold_data
            
            # 2. Riuniamo temporaneamente X e y perché OP4 richiede il dataframe intero
            df_train_fold = pd.concat([X_train_fold, y_train_fold], axis=1)
            df_test_fold = pd.concat([X_test_fold, y_test_fold], axis=1)
            
            # 3. Imputazione valori nulli (OP4)
            print(f"   -> [OP4] Imputazione valori nulli sul Fold {i+1}...")
            # Pulisci il train di questo fold
            risultato_train_fold = run_handle_null_values(df_train_fold)
            df_train_imputato = risultato_train_fold.df_output
            
            dict_output_path = repository_dictionary / f"kfold_{numero_fold}_probability_dictionaries_train.json"
            with open(dict_output_path, "w", encoding="utf-8") as f:
                json.dump(risultato_train_fold.probability_dictionaries, f, indent=4, ensure_ascii=False)
            print(f"Dizionari delle probabilità salvati in: {dict_output_path}")
            
            # Pulisci il test di questo fold
            risultato_test_fold_imputato = run_handle_null_values(df_test_fold)
            df_test_imputato = risultato_test_fold_imputato.df_output
            
            dict_output_path = repository_dictionary / f"kfold_{numero_fold}_probability_dictionaries_test.json"
            with open(dict_output_path, "w", encoding="utf-8") as f:
                json.dump(risultato_test_fold_imputato.probability_dictionaries, f, indent=4, ensure_ascii=False)
            print(f"Dizionari delle probabilità salvati in: {dict_output_path}") 
            

            # 4. Calcolo costi e nomi (OP5) all'interno del fold
            print(f"   -> [OP5] Calcolo costi sul Fold {i+1}...")
            df_train_finale = run_op5(df_train_imputato).df_output
            df_test_finale = run_op5(df_test_imputato).df_output

            # --- A QUESTO PUNTO I DATI DEL FOLD SONO PRONTI PER L'ADDESTRAMENTO ---
            train_out_path = processed_folds_path / f"kfold_{numero_fold}_train_processed.csv"
            test_out_path = processed_folds_path / f"kfold_{numero_fold}_test_processed.csv"
            
            df_train_finale.to_csv(train_out_path, index=False)
            df_test_finale.to_csv(test_out_path, index=False)
            
            # 5. Separazione finale di X e y per addestrare il modello
            X_train_clean = df_train_finale.drop(target, axis=1)
            y_train_clean = df_train_finale[target]

            X_test_clean = df_test_finale.drop(target, axis=1)
            y_test_clean = df_test_finale[target]

            print(f"   -> Addestramento modello sul Fold {i+1}... (Placeholder)")
            
            # QUI INSERIRAI IL TUO MODELLO
            # es: modello.fit(X_train_clean, y_train_clean)
            # es: score = modello.score(X_val_clean, y_val_clean)
            # risultati_addestramento.append(score)
            
               # OP2: Valutazione finale (sui dati completi, puliti e arricchiti)
            print("\n[Esecuzione OP2] Valutazione finale dataset...")
            valutazione = run_evaluation(df_train_finale)  # Puoi anche valutare su df_test_finale se vuoi vedere la performance sul test
            print("Valutazione completata.")
            
            results.append((X_train_clean, X_test_clean, y_train_clean, y_test_clean))
            
        # Per far proseguire la pipeline esplorativa a valle (OP6 e salvataggio),
        # usiamo l'ultimo dataset di addestramento elaborato come rappresentativo.

    
        # OP6: Matrice di correlazione
        print("\n[Esecuzione OP6] Calcolo matrice di correlazione...")
        risultato_op6 = run_op6(
            mio_dataframe_modificato,
            config={"output_dir": "outputs/op6"}
        )
        print("OP6 completato.")

        # if risultato_op6["corr_with_target"] is not None:
        #     print("\nTop correlazioni con Transported:")
        #     print(risultato_op6["corr_with_target"].head(10))  

        return results
    
    elif scelta == '3':
        # OP4 (train): Pulizia iniziale (CryoSleep) e Imputazione valori nulli (gruppi multipli e singoli)
        print("\n[Esecuzione OP4] Gestione e imputazione valori nulli...")
        mio_dataframe_modificato2 = run_handle_null_values(mio_dataframe_modificato)
        mio_dataframe_imputato = mio_dataframe_modificato2.df_output

        # Salvataggio dizionari delle probabilità (basati su OP4)
        dict_output_path = repository_dictionary / "probability_dictionaries.json"
        with open(dict_output_path, "w", encoding="utf-8") as f:
            json.dump(mio_dataframe_modificato2.probability_dictionaries, f, indent=4, ensure_ascii=False)
        print(f"Dizionari delle probabilità salvati in: {dict_output_path}")
        
        print("\n[Esecuzione OP5] Calcolo costi totali e gestione nomi...")
        risultato_op5 = run_op5(mio_dataframe_imputato)
        mio_dataframe_finale = risultato_op5.df_output
        
        # Salvataggio del dataset finale elaborato
        train_out_path = processed_folds_path / "processed_train.csv"
        
        mio_dataframe_finale.to_csv(train_out_path, index=False)
        
        # OP2: Valutazione finale (sui dati completi, puliti e arricchiti)
        print("\n[Esecuzione OP2] Valutazione finale dataset...")
        valutazione = run_evaluation(mio_dataframe_finale)  # Puoi anche valutare su df_test se vuoi vedere la performance sul test
        print("Valutazione completata.")
        
        print(f"\nn[FINE] Dataset finale salvato")   
        # OP6: Matrice di correlazione
        print("\n[Esecuzione OP6] Calcolo matrice di correlazione...")
        risultato_op6 = run_op6(
            mio_dataframe_finale,
            config={"output_dir": "outputs/op6"}
        )
        print("OP6 completato.")
        
        return mio_dataframe_finale

if __name__ == "__main__":
    main()

















