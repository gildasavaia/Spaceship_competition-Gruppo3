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
from op8_scaling import run_scaling 
from op9_encoding import run_encoding

def main():
    print("INIZIO PIPELINE DI PREPROCESSING")
    
    target = "Transported"  # Colonna target per la previsione
    
    # Percorso al dataset originale
    data_path = Path(__file__).resolve().parents[1] / "data" / "train.csv"
    data_path_test = Path(__file__).resolve().parents[1] / "data" / "test.csv"

    # OP1: Lettura
    print("\n[Esecuzione OP1] Lettura dataset...")
    risultato = run_load_and_convert_to_csv(str(data_path))
    mio_dataframe = risultato.df_output
    risultato_test = run_load_and_convert_to_csv(str(data_path_test))
    mio_dataframe_test = risultato_test.df_output
    

    # OP3: Split / feature engineering
    print("\n[Esecuzione OP3] Split dataset e feature engineering...")
    risultato_split = run_split_dataset(mio_dataframe, i_train=1)
    mio_dataframe_modificato = risultato_split.df_output
    risultato_split_test = run_split_dataset(mio_dataframe_test, i_train=0)
    mio_dataframe_modificato_test = risultato_split_test.df_output
    
    # --- PRIMA SCELTA: Metodo di Split ---
    print("\nScegli il metodo di divisione del dataset:")
    print("1: Hold-out (Train/Test split)")
    print("2: K-Fold Cross Validation")
    print("3: Per competizione (Kaggle)")
    scelta = input("Inserisci 1 o 2 o 3: ").strip()
    
    while scelta not in ['1', '2', '3']:
        print("Scelta non valida. Inserisci 1 o 2 o 3.") 
        scelta = input("Inserisci 1 o 2 o 3: ").strip()

    # --- SECONDA SCELTA: Tipo di Modello ---
    print("\nScegli il tipo di modello per cui preparare i dati:")
    print("1: Alberi Decisionali (XGBoost, CATboost, ecc.)")
    print("2: Reti Neurali / Modelli Lineari (Deep Learning, Regressione Logistica, ecc.)")
    scelta_modello = input("Inserisci 1 o 2: ").strip()
    
    while scelta_modello not in ['1', '2']:
        print("Scelta non valida. Inserisci 1 o 2.")
        scelta_modello = input("Inserisci 1 o 2: ").strip()
            
    
    # Creazione cartelle
    repository_data = data_path.parent
    processed_folds_path = repository_data / "preprocessed_folds"
    processed_folds_path.mkdir(parents=True, exist_ok=True)
    repository_dictionary = data_path.parent / "probability_dictionaries"
    repository_dictionary.mkdir(parents=True, exist_ok=True)


    if scelta == '1':
        print("\n[Esecuzione OP7] Divisione Hold-out...")
        X_train, X_test, y_train, y_test = run_holdout(mio_dataframe_modificato)
        
        # OP4 (train)
        print("\n[Esecuzione OP4] Gestione e imputazione valori nulli (Train)...")
        risultato_imputazione = run_handle_null_values(X_train)
        mio_dataframe_imputato = risultato_imputazione.df_output
        
        train_dicts               = risultato_imputazione.probability_dictionaries
        train_age_global_median   = risultato_imputazione.age_global_median
        train_age_hp_medians      = risultato_imputazione.age_homeplanet_medians
        train_age_deck_medians    = risultato_imputazione.age_deck_medians
        train_nz_model            = risultato_imputazione.numzone_model
        train_nz_encoder          = risultato_imputazione.numzone_encoder

        dict_output_path = repository_dictionary / "holdout_probability_dictionaries_train.json"
        with open(dict_output_path, "w", encoding="utf-8") as f:
            json.dump(risultato_imputazione.probability_dictionaries, f, indent=4, ensure_ascii=False)

        # OP5 (train)
        print("\n[Esecuzione OP5] Calcolo costi totali e gestione nomi (Train)...")
        X_train = run_op5(mio_dataframe_imputato).df_output
        
        # OP4 (test)
        print("\n[Esecuzione OP4] Gestione e imputazione valori nulli (Test)...")
        risultato_imputazione = run_handle_null_values(
            X_test,
            train_prob_dicts=train_dicts,
            train_age_global_median=train_age_global_median,
            train_age_homeplanet_medians=train_age_hp_medians,
            train_age_deck_medians=train_age_deck_medians,
            train_numzone_model=train_nz_model,
            train_numzone_encoder=train_nz_encoder,
        )
        mio_dataframe_imputato = risultato_imputazione.df_output

        dict_output_path = repository_dictionary / "holdout_probability_dictionaries_test.json"
        with open(dict_output_path, "w", encoding="utf-8") as f:
            json.dump(risultato_imputazione.probability_dictionaries, f, indent=4, ensure_ascii=False)

        # OP5 (test)
        print("\n[Esecuzione OP5] Calcolo costi totali e gestione nomi (Test)...")
        X_test = run_op5(mio_dataframe_imputato).df_output
        
        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)
        
        # --- OP8: SCALING ---
        print("\n[Esecuzione OP8] Standardizzazione e Log Transformation...")
        scaling_result = run_scaling(df_train, df_test)
        
        # --- OP9: ENCODING ---
        print("\n[Esecuzione OP9] Encoding variabili categoriche...")
        encoding_result = run_encoding(
            scaling_result.df_tree_train,
            scaling_result.df_nn_train,
            scaling_result.df_tree_test,
            scaling_result.df_nn_test
        )
        
        # Salvataggio dinamico basato sulla scelta
        if scelta_modello == '1':
            print("\nSalvataggio versione Alberi Decisionali...")
            encoding_result.df_tree_train.to_csv(processed_folds_path / "holdout_tree_train.csv", index=False)
            encoding_result.df_tree_test.to_csv(processed_folds_path / "holdout_tree_test.csv", index=False)
            df_eval_train = encoding_result.df_tree_train
        else:
            print("\nSalvataggio versione Reti Neurali...")
            encoding_result.df_nn_train.to_csv(processed_folds_path / "holdout_nn_train.csv", index=False)
            encoding_result.df_nn_test.to_csv(processed_folds_path / "holdout_nn_test.csv", index=False)
            df_eval_train = encoding_result.df_nn_train
        
        print("\n[Esecuzione OP2] Valutazione finale dataset...")
        run_evaluation(df_eval_train) 
        
        # print("\n[Esecuzione OP6] Calcolo matrice di correlazione...")
        # run_op6(df_eval_train, output_dir="outputs/op6")
        
        return encoding_result
        
            
    elif scelta == '2':
        print("\n[OP7] Esecuzione K-Fold...")
        folds = run_kfold(mio_dataframe_modificato)
        results = [] 

        for i, fold_data in enumerate(folds):
            print(f"\n--- Elaborazione Fold {i+1} ---")
            numero_fold = i + 1
            
            X_train_fold, X_test_fold, y_train_fold, y_test_fold = fold_data
            
            df_train_fold = pd.concat([X_train_fold, y_train_fold], axis=1)
            df_test_fold = pd.concat([X_test_fold, y_test_fold], axis=1)
            
            # OP4
            print(f"   -> [OP4] Imputazione valori nulli sul Fold {i+1}...")
            risultato_train_fold = run_handle_null_values(df_train_fold)
            df_train_imputato = risultato_train_fold.df_output
            
            train_dicts_fold            = risultato_train_fold.probability_dictionaries
            train_age_global_median_fold = risultato_train_fold.age_global_median
            train_age_hp_medians_fold   = risultato_train_fold.age_homeplanet_medians
            train_age_deck_medians_fold = risultato_train_fold.age_deck_medians
            train_nz_model_fold         = risultato_train_fold.numzone_model
            train_nz_encoder_fold       = risultato_train_fold.numzone_encoder
            
            with open(repository_dictionary / f"kfold_{numero_fold}_prob_dict_train.json", "w", encoding="utf-8") as f:
                json.dump(risultato_train_fold.probability_dictionaries, f, indent=4)
            
            risultato_test_fold = run_handle_null_values(
                df_test_fold,
                train_prob_dicts=train_dicts_fold,
                train_age_global_median=train_age_global_median_fold,
                train_age_homeplanet_medians=train_age_hp_medians_fold,
                train_age_deck_medians=train_age_deck_medians_fold,
                train_numzone_model=train_nz_model_fold,
                train_numzone_encoder=train_nz_encoder_fold,
            )
            df_test_imputato = risultato_test_fold.df_output
            
            with open(repository_dictionary / f"kfold_{numero_fold}_prob_dict_test.json", "w", encoding="utf-8") as f:
                json.dump(risultato_test_fold.probability_dictionaries, f, indent=4)
            
            # OP5
            print(f"   -> [OP5] Calcolo costi sul Fold {i+1}...")
            df_train_finale = run_op5(df_train_imputato).df_output
            df_test_finale = run_op5(df_test_imputato).df_output

            # --- OP8: SCALING ---
            print(f"   -> [OP8] Scaling sul Fold {i+1}...")
            scaling_result = run_scaling(df_train_finale, df_test_finale)

            # --- OP9: ENCODING ---
            print(f"   -> [OP9] Encoding sul Fold {i+1}...")
            encoding_result = run_encoding(
                scaling_result.df_tree_train,
                scaling_result.df_nn_train,
                scaling_result.df_tree_test,
                scaling_result.df_nn_test
            )
            
            # Salvataggio dinamico basato sulla scelta
            if scelta_modello == '1':
                print(f"   -> Salvataggio versione Alberi Decisionali per il Fold {i+1}...")
                encoding_result.df_tree_train.to_csv(processed_folds_path / f"kfold_{numero_fold}_tree_train.csv", index=False)
                encoding_result.df_tree_test.to_csv(processed_folds_path / f"kfold_{numero_fold}_tree_test.csv", index=False)
                df_eval_train = encoding_result.df_tree_train
            else:
                print(f"   -> Salvataggio versione Reti Neurali per il Fold {i+1}...")
                encoding_result.df_nn_train.to_csv(processed_folds_path / f"kfold_{numero_fold}_nn_train.csv", index=False)
                encoding_result.df_nn_test.to_csv(processed_folds_path / f"kfold_{numero_fold}_nn_test.csv", index=False)
                df_eval_train = encoding_result.df_nn_train

            results.append(encoding_result)
            
            if i == len(folds) - 1:
                print("\n[Esecuzione OP2] Valutazione finale sull'ultimo fold...")
                run_evaluation(df_eval_train)
                
                # print("\n[Esecuzione OP6] Calcolo matrice di correlazione sull'ultimo fold...")
                # run_op6(df_eval_train, output_dir="outputs/op6")

        return results
    
    elif scelta == '3':
        # ESECUZIONE FANTASMA: Intero Dataset
        print("\n[Modalità Sviluppatore: Esecuzione su Intero Dataset]")
        
        # OP4 (train completo)
        print("\n[Esecuzione OP4] Gestione e imputazione valori nulli (Train completo)...")
        mio_dataframe_modificato2 = run_handle_null_values(mio_dataframe_modificato)
        mio_dataframe_imputato = mio_dataframe_modificato2.df_output

        dict_output_path = repository_dictionary / "full_probability_dictionaries.json"
        with open(dict_output_path, "w", encoding="utf-8") as f:
            json.dump(mio_dataframe_modificato2.probability_dictionaries, f, indent=4, ensure_ascii=False)

        # OP4 (test Kaggle) — usa modello fittato sul train completo
        print("\n[Esecuzione OP4] Gestione e imputazione valori nulli (Test Kaggle)...")
        mio_dataframe_modificato2_test = run_handle_null_values(
            mio_dataframe_modificato_test,
            train_prob_dicts=mio_dataframe_modificato2.probability_dictionaries,
            train_age_global_median=mio_dataframe_modificato2.age_global_median,
            train_age_homeplanet_medians=mio_dataframe_modificato2.age_homeplanet_medians,
            train_age_deck_medians=mio_dataframe_modificato2.age_deck_medians,
            train_numzone_model=mio_dataframe_modificato2.numzone_model,
            train_numzone_encoder=mio_dataframe_modificato2.numzone_encoder,
        )
        mio_dataframe_imputato_test = mio_dataframe_modificato2_test.df_output
        
        # OP5
        print("\n[Esecuzione OP5] Calcolo costi totali e gestione nomi...")
        mio_dataframe_finale = run_op5(mio_dataframe_imputato).df_output
        mio_dataframe_finale_test = run_op5(mio_dataframe_imputato_test).df_output
        
        
        # OP8
        print("\n[Esecuzione OP8] Scaling dei dati...")
        scaling_result = run_scaling(mio_dataframe_finale)
        scaling_result_test = run_scaling(mio_dataframe_finale_test)

        # OP9
        print("\n[Esecuzione OP9] Encoding variabili categoriche...")
        # Fit sul train e transform sul test (passiamo test come 3°/4° argomento)
        encoding_result = run_encoding(
            scaling_result.df_tree_train,
            scaling_result.df_nn_train,
            scaling_result_test.df_tree_train,
            scaling_result_test.df_nn_train
        )
        # Salvataggio dinamico
        if scelta_modello == '1':
            print("\nSalvataggio versione Alberi Decisionali...")
            encoding_result.df_tree_train.to_csv(processed_folds_path / "processed_full_tree.csv", index=False)
            df_eval_train = encoding_result.df_tree_train
            encoding_result.df_tree_test.to_csv(processed_folds_path / "processed_full_tree_test.csv", index=False)
            
        else:
            print("\nSalvataggio versione Reti Neurali...")
            encoding_result.df_nn_train.to_csv(processed_folds_path / "processed_full_nn.csv", index=False)
            df_eval_train = encoding_result.df_nn_train
            encoding_result.df_nn_test.to_csv(processed_folds_path / "processed_full_nn_test.csv", index=False) 
        print("\n[Esecuzione OP2] Valutazione finale dataset...")
        run_evaluation(df_eval_train) 
        
        # print("\n[Esecuzione OP6] Calcolo matrice di correlazione...")
        run_op6(df_eval_train, output_dir="outputs/op6")
        
        return encoding_result

if __name__ == "__main__":
    main()