from Model_XGboost import *

import os
import glob

# ==========================================
# 🔹 MENU INTERATTIVO
# ==========================================

print("Seleziona il metodo di addestramento per XGBoost:")
print("1. Holdout (singolo train/test)")
print("2. K-Fold (validazione incrociata su più file)")

scelta = input("Inserisci 1 o 2: ").strip()

data_dir = "../data/preprocessed_folds/"




if scelta == "1":

    print("\n Avvio addestramento HOLDOUT...\n")



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

    # Label test
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

        accuracy = evaluate_on_test(
            model,
            X_test,
            y_test
        )

        print(f"\n Accuracy Holdout: {accuracy:.4f}")



    predictions = predict(model, X_test)



    show_predictions(predictions, n=10)




elif scelta == "2":

    print("\n Ricerca dei file K-Fold in corso...")

    # Cerca automaticamente i fold
    search_pattern = os.path.join(
        data_dir,
        "kfold_*_tree_train.csv"
    )

    train_files = glob.glob(search_pattern)

    if not train_files:

        print(" Nessun file K-Fold trovato!")

    else:

        num_folds = len(train_files)

        print(f" Trovati {num_folds} fold.")
        print(" Avvio K-Fold Validation...\n")

        fold_accuracies = []

        for i in range(1, num_folds + 1):

            print(f"\n{'-' * 45}")
            print(f" Fold {i}/{num_folds}")
            print(f"{'-' * 45}")

            train_path = os.path.join(
                data_dir,
                f"kfold_{i}_tree_train.csv"
            )

            test_path = os.path.join(
                data_dir,
                f"kfold_{i}_tree_test.csv"
            )

            # Controllo esistenza test fold
            if not os.path.exists(test_path):

                print(
                    f" File test mancante "
                    f"per fold {i}"
                )

                continue



            train_df, test_df = load_data(
                train_path,
                test_path
            )




            X, y = prepare_data(train_df)
            X_test = prepare_test(test_df)

            # Label test
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

                acc = evaluate_on_test(
                    model,
                    X_test,
                    y_test
                )

                fold_accuracies.append(acc)

                print(
                    f" Accuracy Fold {i}: "
                    f"{acc:.4f}"
                )



            predictions = predict(
                model,
                X_test
            )



        if fold_accuracies:

            media_acc = (
                sum(fold_accuracies)
                / len(fold_accuracies)
            )

            print(f"\n{'=' * 55}")
            print(
                f" Accuracy Media Finale "
                f"su {len(fold_accuracies)} fold: "
                f"{media_acc:.4f}"
            )
            print(f"{'=' * 55}")



else:

    print(
        " Scelta non valida. "
        "Riavvia lo script."
    )