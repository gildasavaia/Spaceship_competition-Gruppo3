from Model_XGboost import *

import os
import glob

from XGBoost.Evaluation_XGBoost import (
    run_full_evaluation,
    print_kfold_summary
)

print("Seleziona il metodo di addestramento per XGBoost:")
print("1. Holdout")
print("2. K-Fold")
print("3. Full Training & Kaggle Submission")
scelta = input("Inserisci 1 o 2 o 3: ").strip()

data_dir = "../data/preprocessed_folds/"


if scelta == "1":

    print("\n Avvio HOLDOUT...\n")

    train_path = os.path.join(
        data_dir,
        "holdout_tree_train.csv"
    )

    test_path = os.path.join(
        data_dir,
        "holdout_tree_test.csv"
    )

    train_df, test_df = load_data(
        train_path,
        test_path
    )

    X, y = prepare_data(train_df)

    X_test = prepare_test(test_df)

    y_test = (
        test_df["Transported"]
        if "Transported" in test_df.columns
        else None
    )

    X = fix_categorical_dtype(X)

    X_test = fix_categorical_dtype(X_test)

    model = create_model()

    train_model(model, X, y)

    if y_test is not None:

        metrics, cm = run_full_evaluation(
            model,
            X_test,
            y_test,
            title="HOLDOUT TEST"
        )

    predictions = predict(
        model,
        X_test
    )

    print("\n Predizioni completate.")


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

        for i in range(1, num_folds + 1):
            # Stampiamo solo una riga veloce per mostrare l'avanzamento senza intasare l'output
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
                # NOTA: Qui impostiamo verbose=False per non stampare i singoli report/grafici
                metrics, cm = run_full_evaluation(
                    model,
                    X_test,
                    y_test,
                    title=f"FOLD {i}",
                    verbose=False
                )
                fold_metrics.append(metrics)
                fold_confusion_matrices.append(cm)

            predictions = predict(model, X_test)

        # Alla fine del ciclo, stampa il summary complessivo con i grafici medi
        if fold_metrics:
            print_kfold_summary(fold_metrics, fold_confusion_matrices)

elif scelta == "3":
    print("\n🚀 Avvio FULL TRAINING XGBoost per Kaggle Submission...\n")

    train_path = os.path.join(data_dir, "processed_full_tree.csv")
    test_path = os.path.join(data_dir, "processed_full_tree_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"❌ Errore: Assicurati che i file 'full_tree' esistano in {data_dir}")
    else:
        # 1. Caricamento
        train_df, test_df = load_data(train_path, test_path)

        # 2. Preparazione
        X_train, y_train = prepare_data(train_df)
        X_test = prepare_test(test_df)

        # 3. Fix categoriche (fondamentale per XGBoost)
        X_train = fix_categorical_dtype(X_train)
        # Nota: X_test verrà sistemato dentro save_submission,
        # ma è bene assicurarsi che non contenga PassengerId qui se prepare_test non lo toglie
        if "PassengerId" in X_test.columns:
            X_test_for_model = X_test.drop("PassengerId", axis=1)
        else:
            X_test_for_model = X_test

        # 4. Training
        model = create_model()
        print("Addestramento finale in corso...")
        train_model(model, X_train, y_train)

        # 5. Salvataggio
        save_submission(model, X_test_for_model, test_df, filename="submission_xgboost_full.csv")

else:
    print(" Scelta non valida.")