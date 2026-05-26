import os
import glob
import numpy as np
import sys
from pathlib import Path

# Importazione controllata dei moduli di definizione e di analisi del modello XGBoost
from Model_XGboost import (
    load_data, prepare_data, prepare_test, create_model,
    fix_categorical_dtype, train_model, predict, save_submission
)
from Evaluation_XGBoost import (
    run_full_evaluation, print_kfold_summary
)

# Registrazione della directory radice nel path per importare l'orchesteratore centralizzato
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from Evaluation.metrics_calculator import MetricsEvaluator

# Gestione delle opzioni utente da riga di comando per l'addestramento dell'algoritmo
print("Seleziona il metodo di addestramento per XGBoost:")
print("1. Holdout")
print("2. K-Fold")
print("3. Full Training & Kaggle Submission")
scelta = input("Inserisci 1 o 2 o 3: ").strip()

data_dir = "../data/preprocessed_folds/"

# =====================================================================
# MODALITÀ 1: CONFIGURAZIONE ED ESECUZIONE HOLDOUT
# =====================================================================
if scelta == "1":
    print("\n Avvio HOLDOUT con XGBoost...\n")
    train_path = os.path.join(data_dir, "holdout_tree_train.csv")
    test_path = os.path.join(data_dir, "holdout_tree_test.csv")

    train_df, test_df = load_data(train_path, test_path)
    X, y = prepare_data(train_df)
    X_test = prepare_test(test_df)

    # Conversione esplicita dei tipi stringa in category per abilitare enable_categorical in XGBoost
    X = fix_categorical_dtype(X)
    X_test = fix_categorical_dtype(X_test)

    y_test = test_df["Transported"] if "Transported" in test_df.columns else None

    # Generazione e addestramento del classificatore ad alberi di decisione stimati via gradiente
    model = create_model()
    train_model(model, X, y)

    if y_test is not None:
        metrics, cm = run_full_evaluation(model, X_test, y_test, title="HOLDOUT TEST (XGBoost)", verbose=True)

    predictions = predict(model, X_test)
    print("\n Predizioni completate.")

    # Consegna formale dei risultati del singolo test all'orchestrazione esterna
    if y_test is not None:
        probabilities = model.predict_proba(X_test)[:, 1]
        valutatore = MetricsEvaluator(y_test, predictions, probabilities, "XGBoost (Holdout)")
        valutatore.export_to_orchestrator()


# =====================================================================
# MODALITÀ 2: VALIDAZIONE INCROCIATA SU K-FOLD
# =====================================================================
elif scelta == "2":
    print("\n Ricerca K-Fold con ottimizzazione iperparametri per XGBoost...\n")
    import random
    from Model_XGboost import create_model_2  # Assicurati di aver aggiunto create_model_2 nel file del modello

    search_pattern = os.path.join(data_dir, "kfold_*_tree_train.csv")
    train_files = glob.glob(search_pattern)

    if not train_files:
        print("Errore: Nessun fold trovato!")
    else:
        num_folds = len(train_files)
        print(f" Trovati {num_folds} fold. Inizio elaborazione...\n")

        # =====================================================================
        # 1. GRIGLIA PARAMETRI E GENERAZIONE COMBINAZIONI CASUALI
        # =====================================================================
        param_grid = {
            "n_estimators": [1500, 2500, 3500, 4500],
            "learning_rate": [0.01, 0.02, 0.05, 0.1],
            "max_depth": [3, 4, 5, 7],
            "min_child_weight": [1, 3, 5],
            "gamma": [0.0, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.85, 0.9, 1.0],
            "reg_lambda": [1, 3, 5, 10],
            "reg_alpha": [0.0, 0.1, 0.3, 0.5]
        }

        # Inseriamo come prima combinazione (Benchmark) i parametri manuali originali
        original_params = {
            "n_estimators": 3500, "learning_rate": 0.02, "max_depth": 5,
            "min_child_weight": 1, "gamma": 0.05, "subsample": 0.9,
            "colsample_bytree": 0.85, "reg_lambda": 3, "reg_alpha": 0.3
        }

        combinazioni = [original_params]
        random.seed(42)  # Seed per garantire la riproducibilità nella scelta delle combinazioni casuali

        # Generiamo le altre 4 combinazioni uniche
        while len(combinazioni) < 5:
            scelta_casuale = {k: random.choice(v) for k, v in param_grid.items()}
            if scelta_casuale not in combinazioni:
                combinazioni.append(scelta_casuale)

        # Variabili di tracciamento per l'ottimizzazione
        best_accuracy = -1
        best_combination_data = {}
        resoconto_classifica = []

        # =====================================================================
        # 2. LOOP DI VALUTAZIONE SULLE 5 COMBINAZIONI DI IPERPARAMETRI
        # =====================================================================
        for idx, params in enumerate(combinazioni):
            is_original = " (PARAMETRI ORIGINALI MANUALI)" if idx == 0 else ""
            print(f"\n" + "=" * 75)
            print(f" TEST COMBINAZIONE {idx + 1}/5{is_original}")
            print(f" Parametri in uso: {params}")
            print("=" * 75)

            fold_metrics = []
            fold_confusion_matrices = []

            # Liste temporanee per accumulare i dati di predizione di questa specifica combinazione
            current_y_true = []
            current_y_pred = []
            current_y_probs = []

            for i in range(1, num_folds + 1):
                train_path = os.path.join(data_dir, f"kfold_{i}_tree_train.csv")
                test_path = os.path.join(data_dir, f"kfold_{i}_tree_test.csv")

                if not os.path.exists(test_path):
                    continue

                train_df, test_df = load_data(train_path, test_path)
                X, y = prepare_data(train_df)
                X_test = prepare_test(test_df)

                X = fix_categorical_dtype(X)
                X_test = fix_categorical_dtype(X_test)

                y_test = test_df["Transported"] if "Transported" in test_df.columns else None

                # Creazione del modello dinamicamente tramite create_model_2
                model = create_model_2(**params)
                train_model(model, X, y)

                if y_test is not None:
                    # verbose=False evita di aprire grafici intermedi fastidiosi
                    metrics, cm = run_full_evaluation(model, X_test, y_test, title=f"FOLD {i}", verbose=False)
                    fold_metrics.append(metrics)
                    fold_confusion_matrices.append(cm)

                predictions = predict(model, X_test)

                if y_test is not None:
                    probabilities = model.predict_proba(X_test)[:, 1]
                    current_y_true.extend(y_test)
                    current_y_pred.extend(predictions)
                    current_y_probs.extend(probabilities)

            # Calcolo delle performance medie del cross-validation per la combinazione corrente
            if fold_metrics:
                avg_acc = np.mean([m['accuracy'] for m in fold_metrics])
                print(f"--> Fine K-Fold. Accuratezza Media Combinazione {idx + 1}: {avg_acc:.4f}")

                resoconto_classifica.append({
                    'index': idx + 1,
                    'is_original': idx == 0,
                    'accuracy': avg_acc
                })

                # Salvataggio dei dati se la combinazione è la migliore trovata finora
                if avg_acc > best_accuracy:
                    best_accuracy = avg_acc
                    best_combination_data = {
                        'params': params,
                        'is_original': idx == 0,
                        'metrics': fold_metrics,
                        'cms': fold_confusion_matrices,
                        'y_true': current_y_true,
                        'y_pred': current_y_pred,
                        'y_probs': current_y_probs
                    }

        # =====================================================================
        # 3. REPORT DI CONFRONTO FINALE
        # =====================================================================
        print("\n\n" + "#" * 70)
        print(" CLASSIFICA FINALE DELLE COMBINAZIONI")
        print("#" * 70)

        # Ordina la classifica per accuratezza decrescente
        for res in sorted(resoconto_classifica, key=lambda x: x['accuracy'], reverse=True):
            tag = " [ORIGINALE PRE-IMPOSTATO]" if res['is_original'] else ""
            print(f" Posizione -> Combinazione {res['index']}{tag}: Accuratezza Media = {res['accuracy']:.4f}")

        print("\n" + "-" * 70)
        if best_combination_data['is_original']:
            print(f" ESITO VERIFICA: I parametri inseriti manualmente a priori\n"
                  f" sono EFFETTIVAMENTE i migliori! (Acc: {best_accuracy:.4f})")
        else:
            print(f" ESITO VERIFICA: Trovata una combinazione casuale MIGLIORE di quella manuale!\n"
                  f" Nuova Miglior Accuratezza: {best_accuracy:.4f}\n"
                  f" Iperparametri vincenti: {best_combination_data['params']}")
        print("-" * 70)

        # =====================================================================
        # 4. ELABORAZIONE DI SINTESI E INVIO ALL'ORCHESTRATORE (Solo per il vincitore)
        # =====================================================================
        print("\n Generazione grafici di sintesi K-Fold per la combinazione vincente...")
        print_kfold_summary(best_combination_data['metrics'], best_combination_data['cms'])

        if best_combination_data['y_true']:
            print("\n Invio definitivo dei dati della combinazione ottimale all'orchestrazione esterna...")
            valutatore = MetricsEvaluator(
                y_true=np.array(best_combination_data['y_true']),
                y_pred=np.array(best_combination_data['y_pred']),
                y_probs=np.array(best_combination_data['y_probs']),
                dataset_name="XGBoost (K-Fold Ottimizzato)"
            )
            valutatore.print_report()
            valutatore.plot_visuals()
            valutatore.export_to_orchestrator()

# =====================================================================
# MODALITÀ 3: ADDESTRAMENTO TOTALE E GENERAZIONE FILE SUBMISSION KAGGLE
# =====================================================================
elif scelta == "3":
    print("\n Avvio FULL TRAINING XGBoost per Kaggle Submission...\n")
    train_path = os.path.join(data_dir, "processed_full_tree.csv")
    test_path = os.path.join(data_dir, "processed_full_tree_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f" Errore: Assicurati che i file 'full_tree' esistano in {data_dir}")
    else:
        train_df, test_df = load_data(train_path, test_path)
        X_train, y_train = prepare_data(train_df)
        X_test = prepare_test(test_df)

        # Rimozione selettiva di PassengerId per evitare disallineamenti nelle feature del modello
        if "PassengerId" in X_test.columns:
            X_test_for_model = X_test.drop("PassengerId", axis=1)
        else:
            X_test_for_model = X_test.copy()

        # Allineamento dei formati prima del fit finale
        X_train = fix_categorical_dtype(X_train)
        X_test_for_model = fix_categorical_dtype(X_test_for_model)

        model = create_model()
        print("Addestramento finale sul dataset completo...")
        train_model(model, X_train, y_train)

        # Scrittura su file dei risultati mappati
        save_submission(model, X_test_for_model, test_df, filename="submission_xgboost_full.csv")

else:
    print(" Scelta non valida.")