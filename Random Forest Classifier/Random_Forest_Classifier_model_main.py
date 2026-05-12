import pandas as pd
import os
import joblib
import warnings
import glob
from pathlib import Path
from Random_Forest_Classifier_model import RandomForestTrainer


# Sostituisci il nome della cartella se l'hai rinominata senza spazi
# (es. from Random_Forest_Classifier_model import ...)

def esegui_pipeline_rf(train_path, test_path, dataset_name, outputs_dir, salva_file_singolo=True):
    """
    Funzione core che esegue l'intera pipeline Random Forest per una specifica coppia di Train/Test.
    Restituisce i DataFrame per permettere l'unione dei K-Fold in un unico file TOTAL.
    """
    print(f"\n{'=' * 60}")
    print(f"ELABORAZIONE RANDOM FOREST: {dataset_name.upper()}")
    print(f"{'=' * 60}")

    # 1. Caricamento Dati
    print("[1/4] Caricamento dati in corso...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=['Transported', 'PassengerId'], errors='ignore')
    y_train = train_df['Transported']
    X_test = test_df.drop(columns=['Transported', 'PassengerId'], errors='ignore')

    # =====================================================================
    # TRUCCO ROBUSTO PER RANDOM FOREST: Unisci, Converti e Dividi
    # Random Forest (sklearn) NON accetta stringhe. Convertiamo tutto in numeri.
    # =====================================================================
    num_train_rows = X_train.shape[0]
    combined_df = pd.concat([X_train, X_test], axis=0, ignore_index=True)

    # Identifichiamo le colonne testuali
    colonne_testuali = combined_df.select_dtypes(include=['object', 'category']).columns.tolist()

    if colonne_testuali:
        print(f"[*] Trasformazione variabili categoriche: {colonne_testuali}")
        # One-Hot Encoding (crea colonne binarie 0/1)
        combined_df = pd.get_dummies(combined_df, columns=colonne_testuali, drop_first=True)

    # Forza tutto il dataset a essere numerico (float)
    combined_df = combined_df.astype(float)

    # Ristacchiamo Train e Test
    X_train = combined_df.iloc[:num_train_rows, :].copy()
    X_test = combined_df.iloc[num_train_rows:, :].copy()
    # =====================================================================

    passenger_ids = test_df['PassengerId'] if 'PassengerId' in test_df.columns else range(len(test_df))

    # 2. Addestramento e Tuning
    print("[2/4] Ricerca iperparametri ottimali (GridSearch)...")
    trainer = RandomForestTrainer(random_state=42)
    trainer.tune_hyperparameters(X_train, y_train)

    # 3. Predizioni e Probabilità
    print("[3/4] Generazione predizioni e probabilità per Stacking...")
    predictions = trainer.predict(X_test)
    probabilities = trainer.predict_proba(X_test)

    # Creazione dei dataframe di output
    res_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Transported': predictions.astype(bool)
    })
    prob_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Probability': probabilities
    })

    # 4. Salvataggio del Modello (.pkl)
    print("[4/4] Salvataggio risultati in 'outputs'...")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    model_file = outputs_dir / f"modello_rf_{dataset_name}.pkl"
    joblib.dump(trainer.best_model, model_file)

    # Salvataggio dei singoli file CSV solo se richiesto
    if salva_file_singolo:
        sub_file = outputs_dir / f"submission_rf_{dataset_name}.csv"
        res_df.to_csv(sub_file, index=False)

        prob_file = outputs_dir / f"prob_rf_{dataset_name}.csv"
        prob_df.to_csv(prob_file, index=False)
        print(f"✅ Modello Random Forest {dataset_name} salvato con successo!")

    return res_df, prob_df


def main():
    print("Avvio della pipeline Random Forest per Spaceship Titanic...\n")

    base_dir = Path(__file__).resolve().parent.parent
    preprocessed_dir = base_dir / "data" / "preprocessed_folds"
    outputs_dir = base_dir / "outputs"

    if not preprocessed_dir.exists():
        print(f"Errore: La cartella {preprocessed_dir} non esiste. Esegui prima la pipeline di preprocessing!")
        return

    # =========================================================
    # MENU INTERATTIVO INTELLIGENTE
    # =========================================================
    print("Seleziona il metodo di addestramento per Random Forest:")
    print("1. Holdout (singolo train/test)")
    print("2. K-Fold (addestramento automatico e output UNIFICATO in file TOTAL)")
    print("3. Processed Full (preparazione della submission finale per Kaggle)")

    scelta = input("Inserisci 1, 2 o 3: ").strip()

    # ---------------------------------
    # OPZIONE 1: HOLDOUT
    # ---------------------------------
    if scelta == "1":
        train_path = preprocessed_dir / "holdout_tree_train.csv"
        test_path = preprocessed_dir / "holdout_tree_test.csv"

        if train_path.exists() and test_path.exists():
            esegui_pipeline_rf(train_path, test_path, "holdout_tree", outputs_dir, salva_file_singolo=True)
        else:
            print("❌ Errore: File holdout mancanti.")

    # ---------------------------------
    # OPZIONE 2: K-FOLD DINAMICO (FILE TOTAL)
    # ---------------------------------
    elif scelta == "2":
        print("\n🔍 Ricerca dei file K-Fold in corso...")
        search_pattern = str(preprocessed_dir / "kfold_*_tree_train.csv")
        train_files = glob.glob(search_pattern)

        if not train_files:
            print("❌ Errore: Nessun file K-Fold trovato nella cartella 'preprocessed_folds'!")
        else:
            num_folds = len(train_files)
            print(f"✅ Trovati {num_folds} fold. Avvio elaborazione in serie...\n")

            all_res = []
            all_prob = []

            for i in range(1, num_folds + 1):
                train_path = preprocessed_dir / f"kfold_{i}_tree_train.csv"
                test_path = preprocessed_dir / f"kfold_{i}_tree_test.csv"

                if not test_path.exists():
                    print(f"⚠️ File test mancante per il fold {i}. Salto...")
                    continue

                # Esegue la pipeline ma NON salva i singoli file
                res, prob = esegui_pipeline_rf(train_path, test_path, f"kfold_{i}_tree", outputs_dir,
                                               salva_file_singolo=False)
                all_res.append(res)
                all_prob.append(prob)

            # --- UNIONE IN UN UNICO FILE ---
            print("\n[*] Unione di tutte le predizioni K-Fold in un unico file TOTAL...")
            final_res = pd.concat(all_res).sort_values('PassengerId')
            final_prob = pd.concat(all_prob).sort_values('PassengerId')

            final_res.to_csv(outputs_dir / "submission_rf_kfold_TOTAL.csv", index=False)
            final_prob.to_csv(outputs_dir / "prob_rf_kfold_TOTAL.csv", index=False)

            print(f"\n🏆 FINE K-FOLD | Tutti i {num_folds} modelli Random Forest sono stati addestrati!")
            print(f"✅ Creato UNICO file di submission: submission_rf_kfold_TOTAL.csv")

    # ---------------------------------
    # OPZIONE 3: FULL KAGGLE DATASET
    # ---------------------------------
    elif scelta == "3":
        train_path = preprocessed_dir / "processed_full_tree.csv"
        test_path = preprocessed_dir / "processed_full_tree_test.csv"

        if train_path.exists() and test_path.exists():
            esegui_pipeline_rf(train_path, test_path, "processed_full_tree", outputs_dir, salva_file_singolo=True)
        else:
            print("❌ Errore: File processed_full mancanti.")

    else:
        print("❌ Scelta non valida. Riavvia lo script.")


if __name__ == "__main__":
    main()