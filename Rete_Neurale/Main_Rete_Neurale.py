from Rete_neurale_model import *
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

data_dir = "../data/preprocessed_folds/"

# ==========================================
# 🔹 OPZIONE 1: HOLDOUT (Tuo codice originale)
# ==========================================
if scelta == "1":
    print("\n🚀 Avvio addestramento con metodo HOLDOUT...\n")

    train_path = f"{data_dir}holdout_nn_train.csv"
    test_path = f"{data_dir}holdout_nn_test.csv"

    # 1. Caricamento dati
    train_df, test_df = load_data(train_path, test_path)

    # 2. Preparazione
    X, y = prepare_data(train_df)
    X_test = prepare_test(test_df)

    # 3. Encoding
    X = pd.get_dummies(X)
    X_test = pd.get_dummies(X_test)
    X_test = X_test.reindex(columns=X.columns, fill_value=0)

    # 4. Creazione modello
    model = create_pipeline_model()

    # 5. FIT PRIMA DI TUTTO
    model = train_model(model, X, y)

    # 6. VALUTAZIONE
    accuracy = evaluate_on_test(model, X, y)
    print(f"\nAccuracy media (NN): {accuracy:.4f}")

    # 7. PREDIZIONE
    predictions = predict(model, X_test)

    # 8. OUTPUT A SCHERMO
    show_predictions(test_df, predictions)

# ==========================================
# 🔹 OPZIONE 2: K-FOLD (Nuova logica dinamica)
# ==========================================
elif scelta == "2":
    print("\n🔍 Ricerca dei file K-Fold in corso...")

    # Usa glob per trovare tutti i file che finiscono con _nn_train.csv (es. kfold_1_nn_train.csv, kfold_2...)
    search_pattern = os.path.join(data_dir, "kfold_*_nn_train.csv")
    train_files = glob.glob(search_pattern)

    if not train_files:
        print("❌ Errore: Nessun file K-Fold trovato nella cartella 'preprocessed_folds'!")
    else:
        num_folds = len(train_files)
        print(f"✅ Trovati {num_folds} fold. Avvio addestramento K-Fold...\n")

        fold_accuracies = []

        # Esegue il ciclo per il numero esatto di fold trovati
        for i in range(1, num_folds + 1):
            print(f"\n{'-' * 40}")
            print(f"🔄 Esecuzione Fold {i}/{num_folds}")
            print(f"{'-' * 40}")

            train_path = os.path.join(data_dir, f"kfold_{i}_nn_train.csv")
            test_path = os.path.join(data_dir, f"kfold_{i}_nn_test.csv")

            if not os.path.exists(test_path):
                print(f"⚠️ File test mancante per il fold {i}. Salto al prossimo...")
                continue

            # Stessa identica logica del tuo codice originale per ogni fold
            train_df, test_df = load_data(train_path, test_path)

            X, y = prepare_data(train_df)
            X_test = prepare_test(test_df)

            X = pd.get_dummies(X)
            X_test = pd.get_dummies(X_test)
            X_test = X_test.reindex(columns=X.columns, fill_value=0)

            model = create_pipeline_model()
            model = train_model(model, X, y)

            # Valutazione (se il test_df contiene le risposte corrette, usa quelle, altrimenti usa il train come nell'holdout)
            if "Transported" in test_df.columns:
                y_test = test_df["Transported"]
                acc = evaluate_on_test(model, X_test, y_test)
            else:
                acc = evaluate_on_test(model, X, y)

            fold_accuracies.append(acc)

        # Calcolo della media finale
        if fold_accuracies:
            media_acc = sum(fold_accuracies) / len(fold_accuracies)
            print(f"\n{'=' * 50}")
            print(f"🏆 FINE K-FOLD | Accuracy Media Finale su {num_folds} fold: {media_acc:.4f}")
            print(f"{'=' * 50}")

else:
    print("❌ Scelta non valida. Esegui di nuovo lo script e inserisci 1 o 2.")