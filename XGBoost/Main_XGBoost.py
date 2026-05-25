from Model_XGboost import *
import os
import glob
import numpy as np  # Ci serve per unire i fold alla fine

from XGBoost.Evaluation_XGBoost import (
    run_full_evaluation,
    print_kfold_summary
)

# =====================================================================
# 🌟 IMPORTIAMO IL VALUTATORE UNIVERSALE PER L'ORCHESTRATORE
# =====================================================================
import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from Evaluation.metrics_calculator import MetricsEvaluator
# =====================================================================

print("Seleziona il metodo di addestramento per XGBoost:")
print("1. Holdout")
print("2. K-Fold")
print("3. Full Training & Kaggle Submission")
scelta = input("Inserisci 1 o 2 o 3: ").strip()

data_dir = "../data/preprocessed_folds/"


if scelta == "1":
    print("\n Avvio HOLDOUT...\n")
    train_path = os.path.join(data_dir, "holdout_tree_train.csv")
    test_path = os.path.join(data_dir, "holdout_tree_test.csv")

    train_df, test_df = load_data(train_path, test_path)
    X, y = prepare_data(train_df)
    X_test = prepare_test(test_df)

    y_test = test_df["Transported"] if "Transported" in test_df.columns else None

    X = fix_categorical_dtype(X)
    X_test = fix_categorical_dtype(X_test)

    model = create_model()
    train_model(model, X, y)

    if y_test is not None:
        metrics, cm = run_full_evaluation(model, X_test, y_test, title="HOLDOUT TEST")

    predictions = predict(model, X_test)
    print("\n Predizioni completate.")

    # 📥 STRADA 2: ESPORTAZIONE AUTOMATICA HOLDOUT PER L'ORCHESTRATORE
    if y_test is not None:
        probabilities = model.predict_proba(X_test)[:, 1]
        valutatore = MetricsEvaluator(y_test, predictions, probabilities, "XGBoost (Holdout)")
        valutatore.export_to_orchestrator()


elif scelta == "2":
    print("\n Ricerca K-Fold...\n")
    search_pattern = os.path.join(data_dir, "kfold_*_tree_train.csv")
    train_files = glob.glob(search_pattern)

    if not train_files:
        print(" Nessun fold trovato!")
    else:
        num_folds = len(train_files)
        print(f" Trovati {num_folds} fold. Elaborazione in corso...\n")

        fold_metrics = []
        fold_confusion_matrices = []

        # LISTE PER RACCOGLIERE I DATI DA MANDARE ALL'ORCHESTRATORE
        all_y_true = []
        all_y_pred = []
        all_y_probs = []

        for i in range(1, num_folds + 1):
            print(f"-> Addestramento ed elaborazione Fold {i}/{num_folds}...")

            train_path = os.path.join(data_dir, f"kfold_{i}_tree_train.csv")
            test_path = os.path.join(data_dir, f"kfold_{i}_tree_test.csv")

            if not os.path.exists(test_path):
                print(f" Fold {i} mancante")
                continue

            train_df, test_df = load_data(train_path, test_path)
            X, y = prepare_data(train_df)
            X_test = prepare_test(test_df)

            y_test = test_df["Transported"] if "Transported" in test_df.columns else None

            X = fix_categorical_dtype(X)
            X_test = fix_categorical_dtype(X_test)

            model = create_model()
            train_model(model, X, y)

            if y_test is not None:
                metrics, cm = run_full_evaluation(model, X_test, y_test, title=f"FOLD {i}", verbose=False)
                fold_metrics.append(metrics)
                fold_confusion_matrices.append(cm)

            predictions = predict(model, X_test)

            # Raccogliamo i dati per l'orchestratore ad ogni fold
            if y_test is not None:
                probabilities = model.predict_proba(X_test)[:, 1]
                all_y_true.extend(y_test)
                all_y_pred.extend(predictions)
                all_y_probs.extend(probabilities)

        if fold_metrics:
            print_kfold_summary(fold_metrics, fold_confusion_matrices)

        # 📥 STRADA 2: ESPORTAZIONE AUTOMATICA K-FOLD GLOBALE PER L'ORCHESTRATORE
        if all_y_true:
            valutatore = MetricsEvaluator(
                y_true=np.array(all_y_true),
                y_pred=np.array(all_y_pred),
                y_probs=np.array(all_y_probs),
                dataset_name="XGBoost (K-Fold)"
            )
            valutatore.print_report()  # Stampa le metriche a schermo
            valutatore.plot_visuals()  # Apre la Matrice di Confusione e la Curva ROC
            valutatore.export_to_orchestrator()  # Manda i voti alla Leaderboard


elif scelta == "3":
    print("\n🚀 Avvio FULL TRAINING XGBoost per Kaggle Submission...\n")
    train_path = os.path.join(data_dir, "processed_full_tree.csv")
    test_path = os.path.join(data_dir, "processed_full_tree_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"❌ Errore: Assicurati che i file 'full_tree' esistano in {data_dir}")
    else:
        # 1. Caricamento dei dati
        train_df, test_df = load_data(train_path, test_path)

        # 2. Separazione in Feature (X) e Target (y)
        X_train, y_train = prepare_data(train_df)
        X_test = prepare_test(test_df)

        # 3. Pulizia di PassengerId se presente
        if "PassengerId" in X_test.columns:
            X_test_for_model = X_test.drop("PassengerId", axis=1)
        else:
            X_test_for_model = X_test.copy()


        X_train = fix_categorical_dtype(X_train)
        X_test_for_model = fix_categorical_dtype(X_test_for_model)

        # 4. Inizializzazione del modello ottimizzato
        model = create_model()

        print("Addestramento finale sul dataset completo in corso...")
        train_model(model, X_train, y_train)

        # 5. Predizione e Salvataggio dei risultati per Kaggle
        save_submission(model, X_test_for_model, test_df, filename="submission_xgboost_full.csv")

else:
    print(" Scelta non valida.")