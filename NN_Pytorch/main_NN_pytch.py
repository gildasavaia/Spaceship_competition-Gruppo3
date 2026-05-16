import os
import glob
import pandas as pd
import torch
import numpy as np  # Ci serve per unire i fold alla fine

# Import dai tuoi file PyTorch
from Model_NN_pytch import load_data, prepare_data, prepare_test, NeuralNet, train_model, predict
from Evaluation_NN_pytch import run_full_evaluation_nn, print_kfold_summary_nn, plot_loss_curve

# =====================================================================
# 🌟 IMPORTIAMO IL VALUTATORE UNIVERSALE PER L'ORCHESTRATORE
# =====================================================================
import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from Evaluation.metrics_calculator import MetricsEvaluator

# =====================================================================

print("Seleziona il metodo di addestramento per la Rete Neurale:")
print("1. Holdout (singolo train/test)")
print("2. K-Fold (validazione incrociata su più file)")
print("3. Full Training & Kaggle Submission")

scelta = input("Inserisci 1 o 2 o 3: ").strip()

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
    print("\n HOLDOUT NN (PyTorch)...\n")

    train_path = os.path.join(data_dir, "holdout_nn_train.csv")
    test_path = os.path.join(data_dir, "holdout_nn_test.csv")

    if not os.path.exists(train_path):
        print(f" Errore: File {train_path} non trovato!")
    else:
        train_df, test_df = load_data(train_path, test_path)

        X, y = prepare_data(train_df)
        X_test = prepare_test(test_df)

        # Allineamento dummies (necessario se ci sono variabili categoriche)
        X, X_test = align_columns(X, X_test)

        # Inizializzazione e addestramento
        model = NeuralNet(input_dim=X.shape[1])
        model = train_model(model, X, y, epochs=50)

        # Valutazione
        if "Transported" in test_df.columns:
            y_test = test_df["Transported"].astype(int)
            metrics, cm = run_full_evaluation_nn(model, X_test, y_test, title="HOLDOUT NN", verbose=True)

        # Predizioni finali e grafici
        predictions = predict(model, X_test)
        plot_loss_curve(model)

        # 📥 STRADA 2: ESPORTAZIONE AUTOMATICA HOLDOUT PER L'ORCHESTRATORE
        if "Transported" in test_df.columns:
            # Ricaviamo le probabilità in modo nativo PyTorch applicando lo scaler prima del passaggio
            model.eval()
            with torch.no_grad():
                X_test_scaled = model.scaler.transform(X_test)
                X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
                probabilities = model(X_tensor).cpu().numpy().flatten()

            valutatore = MetricsEvaluator(
                y_true=y_test.values,
                y_pred=predictions,
                y_probs=probabilities,
                dataset_name="Neural Network (Holdout)"
            )
            valutatore.export_to_orchestrator()


# =========================================================
# 🔹 K-FOLD
# =========================================================
elif scelta == "2":
    print("\n🔍 K-FOLD NN (PyTorch)...\n")

    search_pattern = os.path.join(data_dir, "kfold_*_nn_train.csv")
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

            train_path = os.path.join(data_dir, f"kfold_{i}_nn_train.csv")
            test_path = os.path.join(data_dir, f"kfold_{i}_nn_test.csv")

            if not os.path.exists(train_path) or not os.path.exists(test_path):
                print(f"Fold {i} mancante, salto...")
                continue

            train_df, test_df = load_data(train_path, test_path)
            X, y = prepare_data(train_df)
            X_test = prepare_test(test_df)

            X, X_test = align_columns(X, X_test)

            # Nuovo modello per ogni fold
            model = NeuralNet(input_dim=X.shape[1])

            # Nota: Nascondiamo l'output del training ad ogni epoca stampato da train_model
            # intercettando il print o semplicemente accettando il counter veloce dei fold
            model = train_model(model, X, y, epochs=30)

            if "Transported" in test_df.columns:
                y_test = test_df["Transported"].astype(int)
                # Passiamo verbose=False per non mostrare report e matrici parziali dei singoli fold
                metrics, cm = run_full_evaluation_nn(model, X_test, y_test, title=f"FOLD {i}", verbose=False)

                fold_metrics.append(metrics)
                fold_confusion_matrices.append(cm)

            predictions = predict(model, X_test)

            # Raccogliamo i dati per l'orchestratore ad ogni fold
            if "Transported" in test_df.columns:
                model.eval()
                with torch.no_grad():
                    X_test_scaled = model.scaler.transform(X_test)
                    X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
                    probabilities = model(X_tensor).cpu().numpy().flatten()

                all_y_true.extend(y_test.values)
                all_y_pred.extend(predictions)
                all_y_probs.extend(probabilities)

        # Riepilogo finale a schermo (Mostra le medie e il grafico riassuntivo)
        if fold_metrics:
            print_kfold_summary_nn(fold_metrics, fold_confusion_matrices)

        # 📥 STRADA 2: ESPORTAZIONE AUTOMATICA K-FOLD GLOBALE PER L'ORCHESTRATORE
        if all_y_true:
            valutatore = MetricsEvaluator(
                y_true=np.array(all_y_true),
                y_pred=np.array(all_y_pred),
                y_probs=np.array(all_y_probs),
                dataset_name="Neural Network (K-Fold)"
            )
            valutatore.export_to_orchestrator()


# =========================================================
# 🔹 3. FULL TRAINING & SUBMISSION
# =========================================================
elif scelta == "3":
    print("\n🚀 Avvio FULL TRAINING Neural Network per Kaggle Submission...\n")

    # Adoperiamo gli stessi identici file completi di XGBoost
    train_path = os.path.join(data_dir, "processed_full_tree.csv")
    test_path = os.path.join(data_dir, "processed_full_tree_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"❌ Errore: Assicurati che i file 'full_tree' esistano in {data_dir}")
    else:
        train_df, test_df = load_data(train_path, test_path)

        X_train, y_train = prepare_data(train_df)
        X_test = prepare_test(test_df)

        # Escludiamo PassengerId prima del processamento dummies/allineamento
        if "PassengerId" in X_test.columns:
            X_test_for_model = X_test.drop("PassengerId", axis=1)
        else:
            X_test_for_model = X_test

        # Allineamento One-Hot Encoding tra Train completo e Test completo
        X_train, X_test_for_model = align_columns(X_train, X_test_for_model)

        # Istanziamo la rete neurale basandoci sul numero di colonne generate
        model = NeuralNet(input_dim=X_train.shape[1])

        print("Addestramento finale sul dataset completo in corso...")
        model = train_model(model, X_train, y_train, epochs=50)

        # Calcoliamo le predizioni finali (0 o 1)
        preds = predict(model, X_test_for_model)

        # Creazione del file di Submission coerente con lo standard Kaggle
        submission = pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "Transported": preds.astype(bool)
        })

        filename = "submission_nn_full.csv"
        submission.to_csv(filename, index=False)
        print(f"\n💾 Submission Rete Neurale salvata con successo in: {filename}")

        # Plot facoltativo della Loss Curve finale
        plot_loss_curve(model)

else:
    print(" Scelta non valida.")