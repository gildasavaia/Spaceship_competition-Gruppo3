import os
import glob
import pandas as pd
import torch

# Import dai tuoi file PyTorch
from Model_NN_pytch import load_data, prepare_data, prepare_test, NeuralNet, train_model, predict
from Evaluation_NN_pytch import run_full_evaluation_nn, print_kfold_summary_nn, plot_loss_curve

print("Seleziona il metodo di addestramento per la Rete Neurale:")
print("1. Holdout (singolo train/test)")
print("2. K-Fold (validazione incrociata su più file)")

scelta = input("Inserisci 1 o 2: ").strip()

data_dir = "../data/preprocessed_folds/"


# =========================================================
# 🔹 FUNZIONE DI SUPPORTO PER ALLINEAMENTO COLONNE
# =========================================================
def align_columns(X, X_test):
    X = pd.get_dummies(X)
    X_test = pd.get_dummies(X_test)
    # Assicura che il test set abbia le stesse colonne del train set
    X_test = X_test.reindex(columns=X.columns, fill_value=0)
    return X, X_test


# =========================================================
# 🔹 HOLDOUT
# =========================================================
if scelta == "1":
    print("\n🚀 HOLDOUT NN (PyTorch)...\n")

    train_path = os.path.join(data_dir, "holdout_nn_train.csv")
    test_path = os.path.join(data_dir, "holdout_nn_test.csv")

    if not os.path.exists(train_path):
        print(f"❌ Errore: File {train_path} non trovato!")
    else:
        train_df, test_df = load_data(train_path, test_path)

        X, y = prepare_data(train_df)
        X_test = prepare_test(test_df)

        # Allineamento dummies (necessario se ci sono variabili categoriche)
        X, X_test = align_columns(X, X_test)

        # Inizializzazione e addestramento
        model = NeuralNet(input_dim=X.shape[1])
        model = train_model(model, X, y, epochs=50)  #

        # Valutazione
        if "Transported" in test_df.columns:
            y_test = test_df["Transported"].astype(int)
            metrics, cm = run_full_evaluation_nn(model, X_test, y_test, title="HOLDOUT NN")  #

        # Predizioni finali e grafici
        predictions = predict(model, X_test)  #
        plot_loss_curve(model)  #

# =========================================================
# 🔹 K-FOLD
# =========================================================
elif scelta == "2":
    print("\n🔍 K-FOLD NN (PyTorch)...\n")

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

            if not os.path.exists(train_path) or not os.path.exists(test_path):
                print(f"⚠️ Fold {i} mancante, salto...")
                continue

            train_df, test_df = load_data(train_path, test_path)
            X, y = prepare_data(train_df)
            X_test = prepare_test(test_df)

            X, X_test = align_columns(X, X_test)

            # Nuovo modello per ogni fold
            model = NeuralNet(input_dim=X.shape[1])
            model = train_model(model, X, y, epochs=30)  #

            if "Transported" in test_df.columns:
                y_test = test_df["Transported"].astype(int)
                metrics, cm = run_full_evaluation_nn(model, X_test, y_test, title=f"FOLD {i}")  #

                fold_metrics.append(metrics)
                fold_confusion_matrices.append(cm)

        # Riepilogo finale
        if fold_metrics:
            print_kfold_summary_nn(fold_metrics, fold_confusion_matrices)  #

else:
    print("❌ Scelta non valida.")