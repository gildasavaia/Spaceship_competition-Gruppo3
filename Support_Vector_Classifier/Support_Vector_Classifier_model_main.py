import pandas as pd
import numpy as np  # Importante per convertire le liste in array
import os
import joblib
import warnings
import glob
from pathlib import Path
from Support_Vector_Classifier_model import SVCTrainer

import sys
from pathlib import Path

# Aggiungiamo la cartella principale al percorso per poter importare il valutatore
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from Evaluation.metrics_calculator import MetricsEvaluator

# Soppressione dei warning per un output pulito
warnings.filterwarnings('ignore')


def esegui_pipeline_svc(train_path, test_path, dataset_name, outputs_dir, salva_file_singolo=True):
    """
    Funzione core che esegue l'intera pipeline SVC per una specifica coppia di Train/Test.
    Restituisce i DataFrame per permettere l'unione dei K-Fold.
    """
    print(f"\n{'=' * 60}")
    print(f"ELABORAZIONE SVC: {dataset_name.upper()}")
    print(f"{'=' * 60}")

    # 1. Caricamento Dati
    print("[1/4] Caricamento dati in corso...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=['Transported', 'PassengerId'], errors='ignore')
    y_train = train_df['Transported']
    X_test = test_df.drop(columns=['Transported', 'PassengerId'], errors='ignore')

    # =====================================================================
    # TRUCCO ROBUSTO: Unisci, Converti e Dividi (One-Hot Encoding)
    # L'SVC richiede dati puramente numerici. Trasformiamo le stringhe.
    # =====================================================================
    num_train_rows = X_train.shape[0]
    combined_df = pd.concat([X_train, X_test], axis=0, ignore_index=True)

    colonne_testuali = combined_df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    if colonne_testuali:
        print(f"[*] Conversione variabili categoriche: {colonne_testuali}")
        combined_df = pd.get_dummies(combined_df, columns=colonne_testuali, drop_first=True)

    combined_df = combined_df.astype(float)
    X_train = combined_df.iloc[:num_train_rows, :].copy()
    X_test = combined_df.iloc[num_train_rows:, :].copy()
    # =====================================================================

    passenger_ids = test_df['PassengerId'] if 'PassengerId' in test_df.columns else range(len(test_df))

    # 2. Addestramento
    print("[2/4] Ottimizzazione iperparametri SVC (GridSearch)...")
    trainer = SVCTrainer(random_state=42)
    trainer.tune_hyperparameters(X_train, y_train)

    # 3. Predizioni
    print("[3/4] Generazione predizioni e probabilità per lo Stacking...")
    predictions = trainer.predict(X_test)
    probabilities = trainer.predict_proba(X_test)

    # =====================================================================
    # 🌟 CALCOLO METRICHE IN TEMPO REALE (SOLO PER HOLDOUT)
    # =====================================================================
    if 'Transported' in test_df.columns:
        valutatore = MetricsEvaluator(
            y_true=test_df['Transported'],
            y_pred=predictions,
            y_probs=probabilities,
            dataset_name="SVC (Holdout)"
        )

        # Stampa le metriche SOLO se stiamo facendo l'Holdout o salvando file singoli
        if salva_file_singolo:
            valutatore.print_report()
            valutatore.plot_visuals()

    # Creazione dei dataframe di output
    res_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Transported': predictions.astype(bool)
    })
    prob_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Probability': probabilities
    })

    # 4. Salvataggio del Modello (questo lo facciamo sempre)
    print("[4/4] Salvataggio risultati in 'outputs'...")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    model_file = outputs_dir / f"modello_svc_{dataset_name}.pkl"
    joblib.dump(trainer.best_model, model_file)

    # Salvataggio dei singoli file CSV solo se richiesto (Holdout e Full)
    if salva_file_singolo:
        sub_file = outputs_dir / f"submission_svc_{dataset_name}.csv"
        res_df.to_csv(sub_file, index=False)

        prob_file = outputs_dir / f"prob_svc_{dataset_name}.csv"
        prob_df.to_csv(prob_file, index=False)
        print(f"✅ Modello SVC {dataset_name} salvato con successo!")

    # Catturiamo le risposte vere direttamente dal dataset di test del fold
    y_test_true = test_df['Transported'] if 'Transported' in test_df.columns else None

    return res_df, prob_df, y_test_true, predictions, probabilities


def main():
    print("Avvio della pipeline Support Vector Classifier per Spaceship Titanic...\n")

    base_dir = Path(__file__).resolve().parent.parent
    preprocessed_dir = base_dir / "data" / "preprocessed_folds"
    outputs_dir = base_dir / "outputs"

    if not preprocessed_dir.exists():
        print(f"Errore: La cartella {preprocessed_dir} non esiste. Esegui prima la pipeline di preprocessing!")
        return

    # =========================================================
    # MENU INTERATTIVO INTELLIGENTE
    # =========================================================
    print("Seleziona il metodo di addestramento per SVC:")
    print("1. Holdout (singolo train/test)")
    print("2. K-Fold (addestramento automatico e continuo su TUTTI i fold trovati)")
    print("3. Processed Full (preparazione della submission finale per Kaggle)")

    scelta = input("Inserisci 1, 2 o 3: ").strip()

    # ---------------------------------
    # OPZIONE 1: HOLDOUT
    # ---------------------------------
    if scelta == "1":
        train_path = preprocessed_dir / "holdout_tree_train.csv"
        test_path = preprocessed_dir / "holdout_tree_test.csv"

        if train_path.exists() and test_path.exists():
            esegui_pipeline_svc(train_path, test_path, "holdout_tree", outputs_dir, salva_file_singolo=True)
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
                res, prob, y_true, preds, probs = esegui_pipeline_svc(
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
                    dataset_name="SVC (K-Fold)"
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
            esegui_pipeline_svc(train_path, test_path, "processed_full_tree", outputs_dir, salva_file_singolo=True)
        else:
            print("❌ Errore: File processed_full mancanti.")

    else:
        print("❌ Scelta non valida. Riavvia lo script.")


if __name__ == "__main__":
    main()