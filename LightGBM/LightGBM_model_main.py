import pandas as pd
import numpy as np  # Importante per convertire le liste in array
import os
import joblib
import warnings
import glob
from pathlib import Path
from LightGBM_model import LightGBMTrainer
import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from Evaluation.metrics_calculator import MetricsEvaluator

# Sopprime gli avvisi di LightGBM ("No further splits with positive gain")
warnings.filterwarnings('ignore', category=UserWarning)


def esegui_pipeline_lightgbm(train_path, test_path, dataset_name, outputs_dir, salva_file_singolo=True):
    """
    Funzione core che esegue l'intera pipeline LightGBM per una specifica coppia di Train/Test.
    Restituisce i DataFrame per permettere l'unione dei K-Fold in un unico file TOTAL.
    """
    print(f"\n{'=' * 60}")
    print(f"INIZIO ELABORAZIONE LIGHTGBM: {dataset_name.upper()}")
    print(f"{'=' * 60}")

    # 1. Caricamento Dati
    print("[1/4] Caricamento dati in corso...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=['Transported', 'PassengerId'], errors='ignore')
    y_train = train_df['Transported']
    X_test = test_df.drop(columns=['Transported', 'PassengerId'], errors='ignore')

    # =====================================================================
    # TRUCCO SUPREMO PER LIGHTGBM: Unisci, Converti e Dividi
    # Garantisce che le categorie siano identiche al 100% tra Train e Test
    # =====================================================================
    num_train_rows = X_train.shape[0]
    combined_df = pd.concat([X_train, X_test], axis=0, ignore_index=True)

    colonne_testuali = combined_df.select_dtypes(include=['object', 'string']).columns.tolist()
    if colonne_testuali:
        print(f"[*] Colonne testuali rilevate: {colonne_testuali}. Conversione in 'category'...")
        for col in colonne_testuali:
            combined_df[col] = combined_df[col].astype('category')

    X_train = combined_df.iloc[:num_train_rows, :].copy()
    X_test = combined_df.iloc[num_train_rows:, :].copy()
    # =====================================================================

    passenger_ids = test_df['PassengerId'] if 'PassengerId' in test_df.columns else range(len(test_df))

    # 2. Addestramento e Ottimizzazione
    print("[2/4] Inizializzazione e tuning del modello LightGBM...")
    trainer = LightGBMTrainer(random_state=42)
    trainer.tune_hyperparameters(X_train, y_train)

    # 3. Predizioni e Probabilità
    print("[3/4] Generazione predizioni e probabilità per lo Stacking...")
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

    if 'Transported' in test_df.columns:
        valutatore = MetricsEvaluator(
            y_true=test_df['Transported'],
            y_pred=predictions,
            y_probs=probabilities,
            dataset_name="LightGBM (Holdout)"
        )

        # Nel K-Fold (salva_file_singolo=False) questo blocco starà in silenzio!
        if salva_file_singolo:
            valutatore.print_report()
            valutatore.plot_visuals()

    # 4. Salvataggio del Modello (.pkl)
    print("[4/4] Salvataggio risultati in 'outputs'...")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    model_file = outputs_dir / f"modello_lightgbm_{dataset_name}.pkl"
    joblib.dump(trainer.best_model, model_file)

    # Salvataggio dei singoli file CSV solo se richiesto (es. Holdout o Processed Full)
    if salva_file_singolo:
        sub_file = outputs_dir / f"submission_lightgbm_{dataset_name}.csv"
        res_df.to_csv(sub_file, index=False)

        prob_file = outputs_dir / f"prob_lightgbm_{dataset_name}.csv"
        prob_df.to_csv(prob_file, index=False)
        print(f"✅ File CSV salvati per {dataset_name}.")

        # Catturiamo le risposte vere direttamente dal dataset di test del fold
    y_test_true = test_df['Transported'] if 'Transported' in test_df.columns else None

    return res_df, prob_df, y_test_true, predictions, probabilities


def main():
    print("Avvio della pipeline LightGBM per Spaceship Titanic...\n")

    base_dir = Path(__file__).resolve().parent.parent
    preprocessed_dir = base_dir / "data" / "preprocessed_folds"
    outputs_dir = base_dir / "outputs"

    if not preprocessed_dir.exists():
        print(f"Errore: La cartella {preprocessed_dir} non esiste. Esegui prima la pipeline di preprocessing!")
        return

    # =========================================================
    # MENU INTERATTIVO INTELLIGENTE
    # =========================================================
    print("Seleziona il metodo di addestramento per LightGBM:")
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
            esegui_pipeline_lightgbm(train_path, test_path, "holdout_tree", outputs_dir, salva_file_singolo=True)
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

            # --- NUOVE LISTE PER CATTURARE LE METRICHE ---
            all_y_true = []
            all_y_pred = []
            all_y_probs = []

            for i in range(1, num_folds + 1):
                train_path = preprocessed_dir / f"kfold_{i}_tree_train.csv"
                test_path = preprocessed_dir / f"kfold_{i}_tree_test.csv"

                if not test_path.exists():
                    print(f"⚠️ File test mancante per il fold {i}. Salto...")
                    continue

                # Ora la funzione restituisce 5 elementi, non più solo 2
                res, prob, y_true, preds, probs = esegui_pipeline_lightgbm(
                    train_path, test_path, f"kfold_{i}_tree", outputs_dir, salva_file_singolo=False
                )

                all_res.append(res)
                all_prob.append(prob)

                # Accodiamo le risposte e le previsioni nella nostra "grande lista"
                if y_true is not None:
                    all_y_true.extend(y_true)
                    all_y_pred.extend(preds)
                    all_y_probs.extend(probs)

            # --- UNIONE IN UN UNICO FILE TOTAL ---
            print("\n[*] Unione di tutte le predizioni K-Fold in un unico file TOTAL...")
            final_res = pd.concat(all_res).sort_values('PassengerId')
            final_prob = pd.concat(all_prob).sort_values('PassengerId')

            final_res.to_csv(outputs_dir / "submission_lightgbm_kfold_TOTAL.csv", index=False)
            final_prob.to_csv(outputs_dir / "prob_lightgbm_kfold_TOTAL.csv", index=False)
            print(f"✅ Creato UNICO file di submission!")

            # =====================================================================
            # 🌟 CALCOLO METRICHE GLOBALI ISPIRATO AD XGBOOST (SENZA PASSENGER_ID)
            # =====================================================================
            print("\n[*] Calcolo delle metriche globali sull'intero K-Fold...")
            if all_y_true:
                valutatore = MetricsEvaluator(
                    y_true=np.array(all_y_true),
                    y_pred=np.array(all_y_pred),
                    y_probs=np.array(all_y_probs),
                    dataset_name="LIGHTGBM K-FOLD (GLOBALE)"
                )
                valutatore.print_report()
                valutatore.plot_visuals()
            else:
                print("⚠️ Dati veri mancanti. Impossibile calcolare le metriche globali.")
            # =====================================================================

    # ---------------------------------
    # OPZIONE 3: FULL KAGGLE DATASET
    # ---------------------------------
    elif scelta == "3":
        train_path = preprocessed_dir / "processed_full_tree.csv"
        test_path = preprocessed_dir / "processed_full_tree_test.csv"

        if train_path.exists() and test_path.exists():
            esegui_pipeline_lightgbm(train_path, test_path, "processed_full_tree", outputs_dir, salva_file_singolo=True)
        else:
            print("❌ Errore: File processed_full mancanti.")

    else:
        print("❌ Scelta non valida. Riavvia lo script.")


if __name__ == "__main__":
    main()