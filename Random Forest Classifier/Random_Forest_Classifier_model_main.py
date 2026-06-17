import pandas as pd
import os
import joblib
import warnings
from pathlib import Path
from Random_Forest_Classifier_model import RandomForestTrainer


# Sostituisci il nome della cartella se l'hai rinominata senza spazi
# (es. from Random_Forest_Classifier_model import ...)

def main():
    print("Avvio della pipeline Random Forest per Spaceship Titanic...")

    # Gestione dinamica dei percorsi
    base_dir = Path(__file__).resolve().parent.parent
    preprocessed_dir = base_dir / "data" / "preprocessed_folds"
    outputs_dir = base_dir / "outputs"

    if not preprocessed_dir.exists():
        print(f"Errore: La cartella {preprocessed_dir} non esiste. Esegui prima la pipeline!")
        return

    # =========================================================
    # 1. RICERCA DINAMICA DEI DATASET
    # =========================================================
    dataset_disponibili = []

    # Cerca Holdout
    if (preprocessed_dir / "holdout_tree_train.csv").exists():
        dataset_disponibili.append("holdout_tree")

    # Cerca K-Folds (ordinandoli numericamente)
    fold_files = list(preprocessed_dir.glob("kfold_*_tree_train.csv"))
    fold_files.sort(key=lambda x: int(x.name.split('_')[1]) if x.name.split('_')[1].isdigit() else 0)

    for f in fold_files:
        dataset_disponibili.append(f.name.replace("_train.csv", ""))

    # Cerca Dataset Intero
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
            n_fold = nome.split('_')[1]
            desc = f"K-Fold numero {n_fold}"
        elif "full" in nome:
            desc = "Intero Dataset (per Submission finale)"
        else:
            desc = nome
        print(f"{i}: {desc}")

    scelta = input(f"\nInserisci un numero (1-{len(dataset_disponibili)}): ").strip()
    while scelta not in mappa_dataset:
        scelta = input("Scelta non valida. Riprova: ").strip()

    dataset_scelto = mappa_dataset[scelta]

    # Caricamento percorsi file
    if dataset_scelto == 'processed_full_tree':
        train_path = preprocessed_dir / f"{dataset_scelto}.csv"
    else:
        train_path = preprocessed_dir / f"{dataset_scelto}_train.csv"
    test_path = preprocessed_dir / f"{dataset_scelto}_test.csv"

    print(f"\n{'=' * 60}")
    print(f"ELABORAZIONE RANDOM FOREST: {dataset_scelto.upper()}")
    print(f"{'=' * 60}")

    # ---------------------------------------------------------
    # 3. CARICAMENTO E PREPARAZIONE DATI
    # ---------------------------------------------------------
    print("[1/4] Caricamento dati in corso...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=['Transported', 'PassengerId'], errors='ignore')
    y_train = train_df['Transported']
    X_test = test_df.drop(columns=['Transported', 'PassengerId'], errors='ignore')

    # =====================================================================
    # TRUCCO ROBUSTO PER RANDOM FOREST: Unisci, Converti e Dividi
    # Random Forest (sklearn) NON accetta stringhe. Convertiamo tutto in numeri.
    # =====================================================================
    num_train_rows = X_train.shape[0]
    combined_df = pd.concat([X_train, X_test], axis=0, ignore_index=True)

    # Identifichiamo le colonne testuali
    colonne_testuali = combined_df.select_dtypes(include=['object', 'category']).columns.tolist()

    if colonne_testuali:
        print(f"[*] Trasformazione variabili categoriche: {colonne_testuali}")
        # One-Hot Encoding (crea colonne binarie 0/1)
        combined_df = pd.get_dummies(combined_df, columns=colonne_testuali, drop_first=True)

    # Forza tutto il dataset a essere numerico (float)
    combined_df = combined_df.astype(float)

    # Ristacchiamo Train e Test
    X_train = combined_df.iloc[:num_train_rows, :].copy()
    X_test = combined_df.iloc[num_train_rows:, :].copy()
    # =====================================================================

    passenger_ids = test_df['PassengerId'] if 'PassengerId' in test_df.columns else range(len(test_df))

    # ---------------------------------------------------------
    # 4. ADDESTRAMENTO E TUNING
    # ---------------------------------------------------------
    print("[2/4] Ricerca iperparametri ottimali (GridSearch)...")
    trainer = RandomForestTrainer(random_state=42)
    trainer.tune_hyperparameters(X_train, y_train)

    # ---------------------------------------------------------
    # 5. PREDIZIONI E PROBABILITÀ
    # ---------------------------------------------------------
    print("[3/4] Generazione predizioni e probabilità per Stacking...")
    predictions = trainer.predict(X_test)
    probabilities = trainer.predict_proba(X_test)

    # ---------------------------------------------------------
    # 6. SALVATAGGIO OUTPUT
    # ---------------------------------------------------------
    print("[4/4] Salvataggio file in 'outputs'...")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # File 1: Sottomissione ufficiale (True/False)
    sub_file = outputs_dir / f"submission_rf_{dataset_scelto}.csv"
    pd.DataFrame({
        'PassengerId': passenger_ids,
        'Transported': predictions.astype(bool)
    }).to_csv(sub_file, index=False)

    # File 2: Probabilità per Stacking (0.0 - 1.0)
    prob_file = outputs_dir / f"prob_rf_{dataset_scelto}.csv"
    pd.DataFrame({
        'PassengerId': passenger_ids,
        'Probability': probabilities
    }).to_csv(prob_file, index=False)

    # File 3: Il modello addestrato (.pkl)
    model_file = outputs_dir / f"modello_rf_{dataset_scelto}.pkl"
    joblib.dump(trainer.best_model, model_file)

    print(f"\n✅ Random Forest completato!")
    print(f"-> File Kaggle: {sub_file.name}")
    print(f"-> File Stacking: {prob_file.name}")
    print(f"-> Modello: {model_file.name}")


if __name__ == "__main__":
    main()