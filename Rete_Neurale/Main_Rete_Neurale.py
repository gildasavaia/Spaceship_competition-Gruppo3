from Rete_Neurale.Evaluation_Rete_neurale import plot_loss_curve
from Rete_neurale_model import *
from Evaluation_Rete_neurale import (
    run_full_evaluation_nn,
    print_kfold_summary_nn
)

import pandas as pd
import os
import glob

# ==========================================
# 🔹 Menu Interattivo
# ==========================================
print("Seleziona il metodo di addestramento per la Rete Neurale:")
print("1. Holdout (singolo train/test)")
print("2. K-Fold (validazione incrociata su più file)")
scelta = input("Inserisci 1 o 2: ").strip()

print("Seleziona il metodo di addestramento per la Rete Neurale:")
print("1. Holdout (singolo train/test)")
print("2. K-Fold (validazione incrociata su più file)")

scelta = input("Inserisci 1 o 2: ").strip()

data_dir = "../data/preprocessed_folds/"


# =========================================================
# 🔹 HOLDOUT
# =========================================================

if scelta == "1":

    print("\n🚀 HOLDOUT NN...\n")

    train_path = f"{data_dir}holdout_nn_train.csv"
    test_path = f"{data_dir}holdout_nn_test.csv"

    train_df, test_df = load_data(train_path, test_path)

    X, y = prepare_data(train_df)
    X_test = prepare_test(test_df)

    X = pd.get_dummies(X)
    X_test = pd.get_dummies(X_test)
    X_test = X_test.reindex(columns=X.columns, fill_value=0)

    model = create_pipeline_model()

    model = train_model(model, X, y)

    # =========================
    #  EVALUATION SERIA
    # =========================

    if "Transported" in test_df.columns:

        y_test = test_df["Transported"]

        metrics, cm = run_full_evaluation_nn(
            model,
            X_test,
            y_test,
            title="HOLDOUT NN"
        )

    predictions = predict(model, X_test)

    plot_loss_curve(model)


# =========================================================
# 🔹 K-FOLD
# =========================================================

elif scelta == "2":

    print("\n🔍 K-FOLD NN...\n")

    search_pattern = os.path.join(data_dir, "kfold_*_nn_train.csv")
    train_files = glob.glob(search_pattern)

    if not train_files:
        print("❌ Nessun fold trovato!")

    else:

        num_folds = len(train_files)

        print(f"✅ Trovati {num_folds} fold.\n")

        fold_metrics = []
        fold_confusion_matrices = []

        for i in range(1, num_folds + 1):

            print(f"\n{'-' * 40}")
            print(f"🔄 Fold {i}/{num_folds}")
            print(f"{'-' * 40}")

            train_path = os.path.join(data_dir, f"kfold_{i}_nn_train.csv")
            test_path = os.path.join(data_dir, f"kfold_{i}_nn_test.csv")

            if not os.path.exists(test_path):
                print(f"⚠️ Fold {i} mancante")
                continue

            train_df, test_df = load_data(train_path, test_path)

            X, y = prepare_data(train_df)
            X_test = prepare_test(test_df)

            X = pd.get_dummies(X)
            X_test = pd.get_dummies(X_test)
            X_test = X_test.reindex(columns=X.columns, fill_value=0)

            model = create_pipeline_model()
            model = train_model(model, X, y)

            if "Transported" in test_df.columns:

                y_test = test_df["Transported"]

                metrics, cm = run_full_evaluation_nn(
                    model,
                    X_test,
                    y_test,
                    title=f"FOLD {i}"
                )

                fold_metrics.append(metrics)
                fold_confusion_matrices.append(cm)

            predictions = predict(model, X_test)

        # =========================
        # 🔥 SUMMARY FINALE
        # =========================

        if fold_metrics:

            print_kfold_summary_nn(
                fold_metrics,
                fold_confusion_matrices
            )

else:
    print("❌ Scelta non valida.")