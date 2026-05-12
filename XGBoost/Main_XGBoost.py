from Model_XGboost import *

import os
import glob
from XGBoost.Evaluation_XGBoost import run_full_evaluation



print("Seleziona il metodo di addestramento per XGBoost:")
print("1. Holdout")
print("2. K-Fold")

scelta = input("Inserisci 1 o 2: ").strip()

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

    # ==========================================
    # 🔹 EVALUATION COMPLETA
    # ==========================================

    if y_test is not None:

        metrics = run_full_evaluation(
            model,
            X_test,
            y_test,
            title="HOLDOUT TEST"
        )

    predictions = predict(model, X_test)


# ==========================================
# 🔹 K-FOLD
# ==========================================

elif scelta == "2":

    print("\n🔍 Ricerca K-Fold...\n")

    search_pattern = os.path.join(data_dir, "kfold_*_tree_train.csv")
    train_files = glob.glob(search_pattern)

    if not train_files:

        print("Nessun fold trovato!")

    else:

        num_folds = len(train_files)

        print(f" Trovati {num_folds} fold.\n")

        fold_metrics = []

        for i in range(1, num_folds + 1):

            print(f"\n{'-' * 45}")
            print(f" Fold {i}/{num_folds}")
            print(f"{'-' * 45}")

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

            # ==========================================
            # 🔹 EVALUATION PER OGNI FOLD
            # ==========================================

            if y_test is not None:

                metrics = run_full_evaluation(
                    model,
                    X_test,
                    y_test,
                    title=f"FOLD {i}"
                )

                fold_metrics.append(metrics)

            predictions = predict(model, X_test)

        # ==========================================
        # 🔹 MEDIA FINALE METRICHE
        # ==========================================

        if fold_metrics:

            print(f"\n{'=' * 55}")
            print(" MEDIA RISULTATI K-FOLD")

            avg = {
                k: sum(m[k] for m in fold_metrics) / len(fold_metrics)
                for k in fold_metrics[0]
            }

            for k, v in avg.items():
                print(f"{k:10}: {v:.4f}")

            print(f"{'=' * 55}")

else:
    print(" Scelta non valida.")