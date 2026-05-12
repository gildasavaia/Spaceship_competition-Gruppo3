from Model_CatBoost import (
    load_data,
    prepare_data,
    prepare_test,
    create_catboost_model,
    train_model,
    evaluate_model,
    predict
)

import os
import glob



print("Seleziona il metodo di addestramento per CatBoost:")
print("1. Holdout (singolo train/test)")
print("2. K-Fold (validazione incrociata su più file)")

scelta = input("Inserisci 1 o 2: ").strip()

data_dir = "../data/preprocessed_folds/"




if scelta == "1":

    print("\n Avvio addestramento con metodo HOLDOUT...\n")

    train_path = os.path.join(data_dir, "holdout_tree_train.csv")
    test_path = os.path.join(data_dir, "holdout_tree_test.csv")



    train_df, test_df = load_data(train_path, test_path)

    X, y = prepare_data(train_df)
    X_test = prepare_test(test_df)



    y_test = test_df["Transported"] if "Transported" in test_df.columns else None


    model = create_catboost_model()



    train_model(model, X, y)



    if y_test is not None:
        accuracy = evaluate_model(model, X_test, y_test)
        print(f"\n Accuracy Holdout: {accuracy:.4f}")



    predictions = predict(model, X_test)

    print("\n Predizioni completate.")




elif scelta == "2":

    print("\n Ricerca dei file K-Fold in corso...")

    # Cerca tutti i train fold
    search_pattern = os.path.join(data_dir, "kfold_*_tree_train.csv")
    train_files = glob.glob(search_pattern)

    if not train_files:
        print(" Nessun file K-Fold trovato!")
    else:

        num_folds = len(train_files)

        print(f" Trovati {num_folds} fold.")
        print(" Avvio K-Fold Validation...\n")

        fold_accuracies = []

        for i in range(1, num_folds + 1):

            print(f"\n{'-' * 40}")
            print(f" Fold {i}/{num_folds}")
            print(f"{'-' * 40}")

            train_path = os.path.join(
                data_dir,
                f"kfold_{i}_tree_train.csv"
            )

            test_path = os.path.join(
                data_dir,
                f"kfold_{i}_tree_test.csv"
            )

            # Controllo esistenza
            if not os.path.exists(test_path):
                print(f" Test fold mancante: {test_path}")
                continue

            # =========================
            # LOAD
            # =========================

            train_df, test_df = load_data(train_path, test_path)

            X, y = prepare_data(train_df)
            X_test = prepare_test(test_df)

            # =========================
            # TEST LABEL
            # =========================

            y_test = (
                test_df["Transported"]
                if "Transported" in test_df.columns
                else None
            )

            # =========================
            # MODEL
            # =========================

            model = create_catboost_model()

            # =========================
            # TRAIN
            # =========================

            train_model(model, X, y)

            # =========================
            # EVALUATION
            # =========================

            if y_test is not None:

                acc = evaluate_model(
                    model,
                    X_test,
                    y_test
                )

                fold_accuracies.append(acc)

                print(f" Accuracy Fold {i}: {acc:.4f}")

            # =========================
            # PREDICTIONS
            # =========================

            predictions = predict(model, X_test)

        # ==========================================
        # MEDIA FINALE
        # ==========================================

        if fold_accuracies:

            media_acc = sum(fold_accuracies) / len(fold_accuracies)

            print(f"\n{'=' * 50}")
            print(
                f" Accuracy Media Finale "
                f"su {len(fold_accuracies)} fold: "
                f"{media_acc:.4f}"
            )
            print(f"{'=' * 50}")

# ==========================================
# 🔹 INPUT NON VALIDO
# ==========================================

else:
    print(" Scelta non valida.")