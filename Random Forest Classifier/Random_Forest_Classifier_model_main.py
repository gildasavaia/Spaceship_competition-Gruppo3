import pandas as pd
import os
import joblib
from pathlib import Path
from Random_Forest_Classifier_model import RandomForestTrainer


def main():
    print("Avvio della pipeline Random Forest per Spaceship Titanic...")

    # Gestione percorsi
    base_dir = Path(__file__).resolve().parent.parent
    preprocessed_dir = base_dir / "data" / "preprocessed_folds"
    outputs_dir = base_dir / "outputs"

    if not preprocessed_dir.exists():
        print(f"Errore: La cartella {preprocessed_dir} non esiste. Esegui prima pipeline.py!")
        return

    # =========================================================
    # 1. RICERCA DINAMICA DEI DATASET (Logica speculare a LightGBM)
    # =========================================================
    dataset_disponibili = []

    # Holdout
    if (preprocessed_dir / "holdout_tree_train.csv").exists():
        dataset_disponibili.append("holdout_tree")

    # K-Folds
    fold_files = list(preprocessed_dir.glob("kfold_*_tree_train.csv"))
    fold_files.sort(key=lambda x: int(x.name.split('_')[1]) if x.name.split('_')[1].isdigit() else 0)

    for f in fold_files:
        dataset_disponibili.append(f.name.replace("_train.csv", ""))

    # Dataset Full
    if (preprocessed_dir / "processed_full_tree.csv").exists():
        dataset_disponibili.append("processed_full_tree")

    if not dataset_disponibili:
        print("Errore: Nessun dataset trovato in data/preprocessed_folds.")
        return

    # =========================================================
    # 2. MENU INTERATTIVO
    # =========================================================
    mappa_dataset = {}
    print("\nScegli quale dataset utilizzare per Random Forest:")

    for i, nome in enumerate(dataset_disponibili, start=1):
        mappa_dataset[str(i)] = nome
        if "holdout" in nome:
            desc = "Holdout (Validazione rapida)"
        elif "kfold" in nome:
            n = nome.split('_')[1]
            desc = f"K-Fold numero {n}"
        elif "full" in nome:
            desc = "Dataset Completo (per Submission)"
        else:
            desc = nome
        print(f"{i}: {desc}")

    scelta = input(f"\nInserisci un numero (1-{len(dataset_disponibili)}): ").strip()

    while scelta not in mappa_dataset:
        scelta = input("Scelta non valida. Riprova: ").strip()

    dataset_scelto = mappa_dataset[scelta]

    # Definizione percorsi file
    if dataset_scelto == 'processed_full_tree':
        train_path = preprocessed_dir / f"{dataset_scelto}.csv"
    else:
        train_path = preprocessed_dir / f"{dataset_scelto}_train.csv"

    test_path = preprocessed_dir / f"{dataset_scelto}_test.csv"

    outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"ELABORAZIONE RANDOM FOREST: {dataset_scelto.upper()}")
    print(f"{'=' * 60}")

    # ---------------------------------------------------------
    # 3. CARICAMENTO DATI
    # ---------------------------------------------------------
    print("[1/4] Caricamento dati...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=['Transported', 'PassengerId'], errors='ignore')
    y_train = train_df['Transported']
    X_test = test_df.drop(columns=['Transported', 'PassengerId'], errors='ignore')

    # Nota: Random Forest di scikit-learn non gestisce le 'category' come LightGBM.
    # Se il tuo preprocessing (op9_encoding.py) ha già convertito tutto in numeri,
    # questo passaggio è neutro. Se ci sono ancora stringhe, vanno gestite.

    if 'PassengerId' in test_df.columns:
        passenger_ids = test_df['PassengerId']
    else:
        passenger_ids = range(len(test_df))

    # ---------------------------------------------------------
    # 4. TRAINING E TUNING
    # ---------------------------------------------------------
    print("[2/4] Inizializzazione Random Forest...")
    trainer = RandomForestTrainer(random_state=42)

    print("[3/4] Ricerca iperparametri ottimali (GridSearch)...")
    trainer.tune_hyperparameters(X_train, y_train)

    # ---------------------------------------------------------
    # 5. PREDIZIONE E SALVATAGGIO
    # ---------------------------------------------------------
    print("[4/4] Salvataggio risultati...")
    predictions = trainer.predict(X_test)

    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Transported': predictions.astype(bool)
    })

    # Salvataggio CSV e PKL (per l'Ensemble futuro)
    sub_file = outputs_dir / f"submission_rf_{dataset_scelto}.csv"
    model_file = outputs_dir / f"modello_rf_{dataset_scelto}.pkl"

    submission.to_csv(sub_file, index=False)
    joblib.dump(trainer.best_model, model_file)

    print(f"\nSuccesso!")
    print(f"-> Predizioni: {sub_file}")
    print(f"-> Modello (Cervello): {model_file}")


if __name__ == "__main__":
    main()