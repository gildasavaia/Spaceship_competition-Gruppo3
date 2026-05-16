import os
import glob
from Model_CatBoost import (
    load_data,
    prepare_data,
    prepare_test,
    create_catboost_model,
    train_model,
    predict,
    save_submission
)
from Evaluation_CatBoost import (
    run_full_evaluation,
    print_kfold_summary
)

print("Seleziona il metodo di addestramento per CatBoost:")
print("1. Holdout")
print("2. K-Fold")
print("3. Full Training & Kaggle Submission")
scelta = input("Inserisci 1 o 2 o 3: ").strip()

data_dir = "../data/preprocessed_folds/"

# =========================================================================
# 1. HOLDOUT
# =========================================================================
if scelta == "1":
    print("\n Avvio HOLDOUT con CatBoost...\n")

    train_path = os.path.join(data_dir, "holdout_tree_train.csv")
    test_path = os.path.join(data_dir, "holdout_tree_test.csv")

    train_df, test_df = load_data(train_path, test_path)
    X, y = prepare_data(train_df)
    X_test = prepare_test(test_df)

    y_test = test_df["Transported"] if "Transported" in test_df.columns else None

    # Nota: CatBoost gestisce nativamente i tipi 'object', non serve fix_categorical_dtype
    model = create_catboost_model()
    train_model(model, X, y)

    if y_test is not None:
        metrics, cm = run_full_evaluation(
            model,
            X_test,
            y_test,
            title="HOLDOUT TEST (CatBoost)",
            verbose=True  # Mostra il report e il grafico per l'holdout singolo
        )

    predictions = predict(model, X_test)
    print("\n Predizioni completate.")

# =========================================================================
# 2. K-FOLD
# =========================================================================
elif scelta == "2":
    print("\n Ricerca K-Fold per CatBoost...\n")

    search_pattern = os.path.join(data_dir, "kfold_*_tree_train.csv")
    train_files = glob.glob(search_pattern)

    if not train_files:
        print("❌ Nessun fold trovato!")
    else:
        num_folds = len(train_files)
        print(f" Trovati {num_folds} fold. Elaborazione silenziosa in corso...\n")

        fold_metrics = []
        fold_confusion_matrices = []

        for i in range(1, num_folds + 1):
            print(f"-> Addestramento ed elaborazione Fold {i}/{num_folds}...")

            train_path = os.path.join(data_dir, f"kfold_{i}_tree_train.csv")
            test_path = os.path.join(data_dir, f"kfold_{i}_tree_test.csv")

            if not os.path.exists(test_path):
                print(f" Fold {i} mancante")
                continue

            train_df, test_df = load_data(train_path, test_path)
            X, y = prepare_data(train_df)
            X_test = prepare_test(test_df)

            y_test = test_df["Transported"] if "Transported" in test_df.columns else None

            model = create_catboost_model()

            # Per evitare che i log di CatBoost intasino lo schermo durante il K-Fold,
            # sovrascriviamo temporaneamente il verbose interno di CatBoost a False (o 0)
            model.set_params(verbose=0)
            train_model(model, X, y)

            if y_test is not None:
                # verbose=False evita i print e i grafici dei singoli fold intermedi
                metrics, cm = run_full_evaluation(
                    model,
                    X_test,
                    y_test,
                    title=f"FOLD {i}",
                    verbose=False
                )
                fold_metrics.append(metrics)
                fold_confusion_matrices.append(cm)

            predictions = predict(model, X_test)

        # Stampa finale riassuntiva dei fold (Medie e Grafico finale)
        if fold_metrics:
            print_kfold_summary(fold_metrics, fold_confusion_matrices)

# =========================================================================
# 3. FULL TRAINING & SUBMISSION
# =========================================================================
elif scelta == "3":
    print("\n🚀 Avvio FULL TRAINING CatBoost per Kaggle Submission...\n")

    train_path = os.path.join(data_dir, "processed_full_tree.csv")
    test_path = os.path.join(data_dir, "processed_full_tree_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"❌ Errore: Assicurati che i file 'full_tree' esistano in {data_dir}")
    else:
        train_df, test_df = load_data(train_path, test_path)

        X_train, y_train = prepare_data(train_df)
        X_test = prepare_test(test_df)

        if "PassengerId" in X_test.columns:
            X_test_for_model = X_test.drop("PassengerId", axis=1)
        else:
            X_test_for_model = X_test

        model = create_catboost_model()
        print("Addestramento finale in corso...")
        train_model(model, X_train, y_train)

        save_submission(model, X_test_for_model, test_df, filename="submission_catboost_full.csv")

else:
    print(" Scelta non valida.")