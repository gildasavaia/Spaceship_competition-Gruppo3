import os
import glob
import pandas as pd
# pyrefly: ignore [missing-import]
import torch
import numpy as np
import sys
from pathlib import Path
# Configurazione del percorso per consentire l'accesso al modulo centrale di calcolo metriche
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
# Importazione selettiva dei moduli logici e di valutazione sviluppati per PyTorch
from Model_NN_pytch import load_data, prepare_data, prepare_test, NeuralNet, train_model, predict
from Evaluation.Evaluation_Unified import  run_full_evaluation, print_kfold_summary, plot_loss_curve


from Evaluation.metrics_calculator import MetricsEvaluator

# Menu interattivo per la definizione del flusso operativo della rete neurale
print("Seleziona il metodo di addestramento per la Rete Neurale:")
print("1. Holdout (singolo train/test)")
print("2. K-Fold (validazione incrociata su più file)")
print("3. Full Training & Kaggle Submission")

scelta = input("Inserisci 1 o 2 o 3: ").strip()
data_dir = "../data/preprocessed_folds/"


def align_columns(X, X_test):
    """
    Allinea le colonne di Train e Test per la rete neurale.
    
    I file _nn provenienti dalla pipeline (op9) sono GIÀ one-hot-encoded.
    Questa funzione si limita a:
    1. Convertire eventuali colonne object residue (sicurezza) con get_dummies
    2. Allineare le colonne del test a quelle del train (aggiungendo zeri dove mancano)
    
    """
    # Verifica se ci sono colonne object residue non ancora encodate
    object_cols_train = X.select_dtypes(include=['object']).columns.tolist()
    object_cols_test = X_test.select_dtypes(include=['object']).columns.tolist()
    
    if object_cols_train:
        print(f"[NN] Colonne object residue trovate nel train: {object_cols_train}. Applico get_dummies...")
        X = pd.get_dummies(X, columns=object_cols_train)
    if object_cols_test:
        X_test = pd.get_dummies(X_test, columns=object_cols_test)

    # Reindicizza il test set inserendo zeri dove mancano colonne presenti nel train set
    X_test = X_test.reindex(columns=X.columns, fill_value=0)
    return X, X_test


# =========================================================
# OPZIONE 1: CONFIGURAZIONE HOLDOUT SINGOLO
# =========================================================
if scelta == "1":
    print("\n HOLDOUT NN (PyTorch)...\n")

    train_path = os.path.join(data_dir, "holdout_nn_train.csv")
    test_path = os.path.join(data_dir, "holdout_nn_test.csv")

    if not os.path.exists(train_path):
        print(f"Errore: Assicurati che i file 'holdout_nn' esistano in {data_dir}")
    else:
        train_df, test_df = load_data(train_path, test_path)

        X, y = prepare_data(train_df)
        X_test = prepare_test(test_df)

        # Allineamento strutturale dei DataFrame prima della standardizzazione
        X, X_test = align_columns(X, X_test)

        # Creazione della rete passando la dimensione dinamica delle feature come input_dim
        model = NeuralNet(input_dim=X.shape[1])
        model = train_model(model, X, y, epochs=50)

        # Calcola la valutazione formale se la colonna target reale esiste
        if "Transported" in test_df.columns:
            y_test = test_df["Transported"].astype(int)
            metrics, cm = run_full_evaluation(model, X_test, y_test, title="HOLDOUT NN", verbose=True, is_nn=True)

        predictions = predict(model, X_test)
        plot_loss_curve(model)

        # Estrazione nativa delle probabilità per la registrazione nell'orchestratore
        if "Transported" in test_df.columns:
            model.eval()
            with torch.no_grad():
                # Applica sui dati di test lo scaler fittato durante il training del modello
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
# OPZIONE 2: VALIDAZIONE INCROCIATA K-FOLD
# =========================================================
elif scelta == "2":
    print("\n K-FOLD NN (PyTorch)\n")

    search_pattern = os.path.join(data_dir, "kfold_*_nn_train.csv")
    train_files = glob.glob(search_pattern)

    if not train_files:
        print(f" Errore: Assicurati che i file 'kfold_*_nn_train.csv' esistano in {data_dir}")
    else:
        num_folds = len(train_files)
        print(f" Trovati {num_folds} fold. Elaborazione in corso...\n")

        fold_metrics = []
        fold_confusion_matrices = []

        # Array globali di raccolta per le metriche aggregate dell'orchestratore
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

            # Istanziazione di un nuovo modello vergine per resettare i pesi ad ogni fold
            model = NeuralNet(input_dim=X.shape[1])
            model = train_model(model, X, y, epochs=30)

            if "Transported" in test_df.columns:
                y_test = test_df["Transported"].astype(int)
                # Esegue la valutazione silenziata impostando verbose=False
                metrics, cm = run_full_evaluation(model, X_test, y_test, title=f"FOLD {i}", verbose=False, is_nn=True)

                fold_metrics.append(metrics)
                fold_confusion_matrices.append(cm)

            predictions = predict(model, X_test)

            # Estrazione delle probabilità del singolo fold per l'unione finale dei dati
            if "Transported" in test_df.columns:
                model.eval()
                with torch.no_grad():
                    X_test_scaled = model.scaler.transform(X_test)
                    X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
                    probabilities = model(X_tensor).cpu().numpy().flatten()

                all_y_true.extend(y_test.values)
                all_y_pred.extend(predictions)
                all_y_probs.extend(probabilities)

        # Stampa il riepilogo complessivo aggregato di tutti i fold elaborati
        if fold_metrics:
            print_kfold_summary(fold_metrics, fold_confusion_matrices, is_nn=True)

        # Esporta l'array unificato contenente i risultati di tutti i fold
        if all_y_true:
            valutatore = MetricsEvaluator(
                y_true=np.array(all_y_true),
                y_pred=np.array(all_y_pred),
                y_probs=np.array(all_y_probs),
                dataset_name="Neural Network (K-Fold)"
            )
            valutatore.export_to_orchestrator()



elif scelta == "3":
    print("\n Avvio FULL TRAINING Neural Network per Kaggle Submission...\n")

    train_path = os.path.join(data_dir, "processed_full_nn.csv")
    test_path = os.path.join(data_dir, "processed_full_nn_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f" Errore: Assicurati che i file 'full_nn' esistano in {data_dir}")
    else:
        train_df, test_df = load_data(train_path, test_path)

        X_train, y_train = prepare_data(train_df)
        X_test = prepare_test(test_df)

        # Rimozoione di PassengerId prima delle dummies per evitare di frammentare la matrice
        if "PassengerId" in X_test.columns:
            X_test_for_model = X_test.drop("PassengerId", axis=1)
        else:
            X_test_for_model = X_test

        X_train, X_test_for_model = align_columns(X_train, X_test_for_model)

        model = NeuralNet(input_dim=X_train.shape[1])

        print("Addestramento finale sul dataset completo in corso...")
        model = train_model(model, X_train, y_train, epochs=200)

        preds = predict(model, X_test_for_model)

        # Costruzione del DataFrame finale formattato e tipizzato in booleano per Kaggle
        submission = pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "Transported": preds.astype(bool)
        })

        # Calcolo dinamico della cartella outputs dalla root del progetto
        outputs_dir = base_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Definizione del percorso completo del file
        filename = "submission_nn_full.csv"
        filepath = outputs_dir / filename
        
        submission.to_csv(str(filepath), index=False)
        print(f"\n Submission Rete Neurale salvata con successo in: {filepath}")
        plot_loss_curve(model)
 
else:
    print(" Scelta non valida.")