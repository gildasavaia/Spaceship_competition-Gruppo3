import os
import glob
import numpy as np
import sys
from pathlib import Path

# Importazione controllata dei moduli di definizione e di analisi del modello XGBoost
from Model_XGboost import (
    load_data, prepare_data, prepare_test, create_model,
    fix_categorical_dtype, train_model, predict, save_submission
)
from Evaluation_XGBoost import (
    run_full_evaluation, print_kfold_summary
)

# Registrazione della directory radice nel path per importare l'orchesteratore centralizzato
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from Evaluation.metrics_calculator import MetricsEvaluator

# Gestione delle opzioni utente da riga di comando per l'addestramento dell'algoritmo
print("Seleziona il metodo di addestramento per XGBoost:")
print("1. Holdout")
print("2. K-Fold")
print("3. Full Training & Kaggle Submission")
scelta = input("Inserisci 1 o 2 o 3: ").strip()

data_dir = "../data/preprocessed_folds/"

# =====================================================================
# MODALITÀ 1: CONFIGURAZIONE ED ESECUZIONE HOLDOUT
# =====================================================================
if scelta == "1":
    print("\n Avvio HOLDOUT con XGBoost...\n")
    train_path = os.path.join(data_dir, "holdout_tree_train.csv")
    test_path = os.path.join(data_dir, "holdout_tree_test.csv")

    train_df, test_df = load_data(train_path, test_path)
    X, y = prepare_data(train_df)
    X_test = prepare_test(test_df)

    # Conversione esplicita dei tipi stringa in category per abilitare enable_categorical in XGBoost
    X = fix_categorical_dtype(X)
    X_test = fix_categorical_dtype(X_test)

    y_test = test_df["Transported"] if "Transported" in test_df.columns else None

    # Generazione e addestramento del classificatore ad alberi di decisione stimati via gradiente
    model = create_model()
    train_model(model, X, y)

    if y_test is not None:
        metrics, cm = run_full_evaluation(model, X_test, y_test, title="HOLDOUT TEST (XGBoost)", verbose=True)

    predictions = predict(model, X_test)
    print("\n Predizioni completate.")

    # Consegna formale dei risultati del singolo test all'orchestrazione esterna
    if y_test is not None:
        probabilities = model.predict_proba(X_test)[:, 1]
        valutatore = MetricsEvaluator(y_test, predictions, probabilities, "XGBoost (Holdout)")
        valutatore.export_to_orchestrator()


# =====================================================================
# MODALITÀ 2: VALIDAZIONE INCROCIATA SU K-FOLD
# =====================================================================
elif scelta == "2":
    print("\n Ricerca K-Fold per XGBoost...\n")
    search_pattern = os.path.join(data_dir, "kfold_*_tree_train.csv")
    train_files = glob.glob(search_pattern)

    if not train_files:
        print("Errore: Nessun fold trovato!")
    else:
        num_folds = len(train_files)
        print(f" Trovati {num_folds} fold. Inizio elaborazione...\n")

        fold_metrics = []
        fold_confusion_matrices = []

        # Contenitori estesi per consolidare le predizioni effettuate su ogni porzione di dati
        all_y_true = []
        all_y_pred = []
        all_y_probs = []

        for i in range(1, num_folds + 1):
            print(f"-> Addestramento ed elaborazione Fold {i}/{num_folds}...")

            train_path = os.path.join(data_dir, f"kfold_{i}_tree_train.csv")
            test_path = os.path.join(data_dir, f"kfold_{i}_tree_test.csv")

            if not os.path.exists(test_path):
                print(f" Fold {i} non presente su disco, salto.")
                continue

            train_df, test_df = load_data(train_path, test_path)
            X, y = prepare_data(train_df)
            X_test = prepare_test(test_df)

            # Sincronizzazione dei tipi categorici obbligatoria per ogni iterazione dei file
            X = fix_categorical_dtype(X)
            X_test = fix_categorical_dtype(X_test)

            y_test = test_df["Transported"] if "Transported" in test_df.columns else None

            model = create_model()
            train_model(model, X, y)

            if y_test is not None:
                # Raccolta asincrona delle metriche escludendo output grafici intermedi
                metrics, cm = run_full_evaluation(model, X_test, y_test, title=f"FOLD {i}", verbose=False)
                fold_metrics.append(metrics)
                fold_confusion_matrices.append(cm)

            predictions = predict(model, X_test)

            if y_test is not None:
                probabilities = model.predict_proba(X_test)[:, 1]
                all_y_true.extend(y_test)
                all_y_pred.extend(predictions)
                all_y_probs.extend(probabilities)

        if fold_metrics:
            print_kfold_summary(fold_metrics, fold_confusion_matrices)

        # Invio definitivo delle liste flat concatenate all'istanza dell'orchestratore universale
        if all_y_true:
            valutatore = MetricsEvaluator(
                y_true=np.array(all_y_true),
                y_pred=np.array(all_y_pred),
                y_probs=np.array(all_y_probs),
                dataset_name="XGBoost (K-Fold)"
            )
            valutatore.print_report()
            valutatore.plot_visuals()
            valutatore.export_to_orchestrator()


# =====================================================================
# MODALITÀ 3: ADDESTRAMENTO TOTALE E GENERAZIONE FILE SUBMISSION KAGGLE
# =====================================================================
elif scelta == "3":
    print("\n Avvio FULL TRAINING XGBoost per Kaggle Submission...\n")
    train_path = os.path.join(data_dir, "processed_full_tree.csv")
    test_path = os.path.join(data_dir, "processed_full_tree_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f" Errore: Assicurati che i file 'full_tree' esistano in {data_dir}")
    else:
        train_df, test_df = load_data(train_path, test_path)
        X_train, y_train = prepare_data(train_df)
        X_test = prepare_test(test_df)

        # Rimozione selettiva di PassengerId per evitare disallineamenti nelle feature del modello
        if "PassengerId" in X_test.columns:
            X_test_for_model = X_test.drop("PassengerId", axis=1)
        else:
            X_test_for_model = X_test.copy()

        # Allineamento dei formati prima del fit finale
        X_train = fix_categorical_dtype(X_train)
        X_test_for_model = fix_categorical_dtype(X_test_for_model)

        model = create_model()
        print("Addestramento finale sul dataset completo...")
        train_model(model, X_train, y_train)

        # Scrittura su file dei risultati mappati
        save_submission(model, X_test_for_model, test_df, filename="submission_xgboost_full.csv")

else:
    print(" Scelta non valida.")