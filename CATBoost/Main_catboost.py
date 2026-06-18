import os
import glob
import numpy as np
import sys
from pathlib import Path
from Model_CatBoost import (
    load_data, prepare_data, prepare_test, create_catboost_model,
    train_model, predict, save_submission
)
from Evaluation_CatBoost import (
    run_full_evaluation, print_kfold_summary
)

# Calcolo dinamico della directory principale per permettere l'importazione di moduli esterni
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from Evaluation.metrics_calculator import MetricsEvaluator

# Interfaccia utente via terminale per la selezione della modalità operativa
print("Seleziona il metodo di addestramento per CatBoost:")
print("1. Holdout")
print("2. K-Fold")
print("3. Full Training & Kaggle Submission")
scelta = input("Inserisci 1 o 2 o 3: ").strip()

# Directory contenente i file CSV generati in fase di preprocessing
data_dir = "../data/preprocessed_folds/"



# =====================================================================
# MODALITÀ 1: VALUTAZIONE TRAMITE HOLDOUT (TRAIN/TEST SINGOLO)
# =====================================================================
if scelta == "1":
    print("\n Avvio HOLDOUT con CatBoost...\n")
    train_path = os.path.join(data_dir, "holdout_tree_train.csv")
    test_path = os.path.join(data_dir, "holdout_tree_test.csv")

    # Caricamento dei dataset e separazione delle feature dal target
    train_df, test_df = load_data(train_path, test_path)
    X, y = prepare_data(train_df)
    X_test = prepare_test(test_df)

    # Estrazione del target reale dal test set se disponibile nel file
    y_test = test_df["Transported"] if "Transported" in test_df.columns else None

    # Inizializzazione del classificatore ed esecuzione dell'addestramento
    model = create_catboost_model()
    train_model(model, X, y)

    # Se il target è presente, esegue la valutazione completa e genera i grafici
    if y_test is not None:
        metrics, cm = run_full_evaluation(model, X_test, y_test, title="HOLDOUT TEST (CatBoost)", verbose=True)

    # Generazione delle predizioni finali sul set di test
    predictions = predict(model, X_test)
    print("\n Predizioni completate.")

    # Esportazione automatica dei vettori di performance verso l'orchestratore centrale
    if y_test is not None:
        probabilities = model.predict_proba(X_test)[:, 1]
        valutatore = MetricsEvaluator(y_test, predictions, probabilities, "CatBoost (Holdout)")
        valutatore.export_to_orchestrator()


# =====================================================================
# MODALITÀ 2: VALIDAZIONE INCROCIATA (K-FOLD CROSS-VALIDATION)
# =====================================================================
elif scelta == "2":
    print("\n Ricerca K-Fold per CatBoost...\n")
    # Identificazione dinamica di tutti i file di train corrispondenti allo schema kfold
    search_pattern = os.path.join(data_dir, "kfold_*_tree_train.csv")
    train_files = glob.glob(search_pattern)

    if not train_files:
        print("Errore: Nessun fold trovato!")
    else:
        num_folds = len(train_files)
        print(f" Trovati {num_folds} fold. Elaborazione in corso...\n")

        # Strutture dati per raccogliere i risultati parziali di ogni singolo fold
        fold_metrics = []
        fold_confusion_matrices = []

        # Liste di accumulo per raggruppare tutte le predizioni della cross-validation
        all_y_true = []
        all_y_pred = []
        all_y_probs = []

        # Iterazione ciclica su ciascun fold identificato
        for i in range(1, num_folds + 1):
            print(f"-> Addestramento ed elaborazione Fold {i}/{num_folds}...")

            train_path = os.path.join(data_dir, f"kfold_{i}_tree_train.csv")
            test_path = os.path.join(data_dir, f"kfold_{i}_tree_test.csv")

            # Salta l'iterazione corrente se il file di test associato al fold non esiste
            if not os.path.exists(test_path):
                print(f" Fold {i} mancante")
                continue

            # Caricamento e preparazione dei dati specifici del fold corrente
            train_df, test_df = load_data(train_path, test_path)
            X, y = prepare_data(train_df)
            X_test = prepare_test(test_df)

            y_test = test_df["Transported"] if "Transported" in test_df.columns else None

            # Configurazione del modello con parametro verbose=0 per evitare log ridondanti nei cicli
            model = create_catboost_model()
            model.set_params(verbose=0)
            train_model(model, X, y)

            # Valutazione silenziosa (verbose=False) per popolare le liste delle metriche
            if y_test is not None:
                metrics, cm = run_full_evaluation(model, X_test, y_test, title=f"FOLD {i}", verbose=False)
                fold_metrics.append(metrics)
                fold_confusion_matrices.append(cm)

            predictions = predict(model, X_test)

            # Aggregazione progressiva dei dati per l'analisi globale post-ciclo
            if y_test is not None:
                probabilities = model.predict_proba(X_test)[:, 1]
                all_y_true.extend(y_test)
                all_y_pred.extend(predictions)
                all_y_probs.extend(probabilities)

        # Stampa il riepilogo complessivo delle medie al termine dell'elaborazione di tutti i fold
        if fold_metrics:
            print_kfold_summary(fold_metrics, fold_confusion_matrices)

        # Trasmissione dei dati complessivi accumulati all'orchestratore per l'aggiornamento della leaderboard
        if all_y_true:
            valutatore = MetricsEvaluator(
                y_true=np.array(all_y_true),
                y_pred=np.array(all_y_pred),
                y_probs=np.array(all_y_probs),
                dataset_name="CatBoost (K-Fold)"
            )
            valutatore.print_report()
            valutatore.plot_visuals()
            valutatore.export_to_orchestrator()


# =====================================================================
# MODALITÀ 3: ADDESTRAMENTO SULL'INTERO DATASET E GENERAZIONE SUBMISSION
# =====================================================================
elif scelta == "3":
    print("\n Avvio FULL TRAINING CatBoost per Kaggle Submission...\n")
    train_path = os.path.join(data_dir, "processed_full_tree.csv")
    test_path = os.path.join(data_dir, "processed_full_tree_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f" Errore: Assicurati che i file 'full_tree' esistano in {data_dir}")
    else:
        # Caricamento del dataset completo (senza split di validazione)
        train_df, test_df = load_data(train_path, test_path)
        X_train, y_train = prepare_data(train_df)
        X_test = prepare_test(test_df)

        # Esclusione dell'indice PassengerId per evitare il binarizzarsi di feature non predittive
        if "PassengerId" in X_test.columns:
            X_test_for_model = X_test.drop("PassengerId", axis=1)
        else:
            X_test_for_model = X_test

        # Addestramento finale ed esportazione del file di sottomissione conforme a Kaggle
        model = create_catboost_model()
        print("Addestramento finale in corso...")
        train_model(model, X_train, y_train)
        save_submission(model, X_test_for_model, test_df, filename="submission_catboost_full.csv")

else:
    print(" Scelta non valida.")