from Model_CatBoost import (
    load_data,
    prepare_data,
    prepare_test,
    create_catboost_model,
    train_model,
    predict,
    save_submission
)

from CATBoost.Evaluation_CatBoost import (
    run_full_evaluation,
    print_kfold_summary
)

import os
import glob

print("Seleziona il metodo di addestramento per CatBoost:")
print("1. Holdout")
print("2. K-Fold")
print("3. Full Training & Kaggle Submission")
scelta = input("Inserisci 1 o 2 o 3: ").strip()

data_dir = "../data/preprocessed_folds/"


if scelta == "1":

    print("\n🚀 Avvio HOLDOUT...\n")

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

    model = create_catboost_model()

    train_model(
        model,
        X,
        y
    )

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

    print("\n✅ Predizioni completate.")


elif scelta == "2":

    print("\n🔍 Ricerca K-Fold...\n")

    search_pattern = os.path.join(
        data_dir,
        "kfold_*_tree_train.csv"
    )

    train_files = glob.glob(search_pattern)

    if not train_files:

        print("❌ Nessun fold trovato!")

    else:

        num_folds = len(train_files)

        print(f"✅ Trovati {num_folds} fold.\n")

        fold_metrics = []

        fold_confusion_matrices = []

        for i in range(1, num_folds + 1):

            print(f"\n{'-' * 45}")
            print(f"🔄 Fold {i}/{num_folds}")
            print(f"{'-' * 45}")

            train_path = os.path.join(
                data_dir,
                f"kfold_{i}_tree_train.csv"
            )

            test_path = os.path.join(
                data_dir,
                f"kfold_{i}_tree_test.csv"
            )

            if not os.path.exists(test_path):

                print(f"⚠️ Fold {i} mancante")

                continue

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

            model = create_catboost_model()

            train_model(
                model,
                X,
                y
            )

            if y_test is not None:

                metrics, cm = run_full_evaluation(
                    model,
                    X_test,
                    y_test,
                    title=f"FOLD {i}"
                )

                fold_metrics.append(metrics)

                fold_confusion_matrices.append(cm)

            predictions = predict(
                model,
                X_test
            )

        if fold_metrics:

            print_kfold_summary(
                fold_metrics,
                fold_confusion_matrices
            )
elif scelta == "3":
    print("\n🚀 Avvio FULL TRAINING per Kaggle Submission...\n")

    # Percorsi per i file "Full"
    train_path = os.path.join(data_dir, "processed_full_tree.csv")
    test_path = os.path.join(data_dir, "processed_full_tree_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"❌ Errore: Assicurati che {train_path} e {test_path} esistano!")
    else:
        # 1. Caricamento dati
        train_df, test_df = load_data(train_path, test_path)

        # 2. Preparazione
        X_train, y_train = prepare_data(train_df)
        X_test = prepare_test(test_df)

        # 3. Training sul dataset completo
        model = create_catboost_model()
        print("Sperimentazione su tutto il dataset di training...")
        train_model(model, X_train, y_train)

        # 4. Generazione Submission
        # Nota: passiamo test_df originale per recuperare i PassengerId
        save_submission(model, X_test, test_df, filename="submission_catboost_full.csv")
else:

    print("❌ Scelta non valida.")