import pandas as pd
import os
import joblib
import warnings
from pathlib import Path
from LightGBM_model import LightGBMTrainer

# Sopprime gli avvisi di LightGBM ("No further splits with positive gain")
warnings.filterwarnings('ignore', category=UserWarning)


def main():
    print("Avvio della pipeline LightGBM per Spaceship Titanic...")

    base_dir = Path(__file__).resolve().parent.parent
    preprocessed_dir = base_dir / "data" / "preprocessed_folds"
    outputs_dir = base_dir / "outputs"

    if not preprocessed_dir.exists():
        print(f"Errore: La cartella {preprocessed_dir} non esiste. Esegui prima pipeline.py!")
        return

    # =========================================================
    # 1. RICERCA DINAMICA DEI DATASET
    # =========================================================
    dataset_disponibili = []

    # Cerca l'Holdout
    if (preprocessed_dir / "holdout_tree_train.csv").exists():
        dataset_disponibili.append("holdout_tree")

    # Cerca tutti i file K-Fold dinamicamente e li ordina per numero
    fold_files = list(preprocessed_dir.glob("kfold_*_tree_train.csv"))
    # Funzione per ordinare i file (estrae il numero dal nome, es. da 'kfold_2' prende 2)
    fold_files.sort(key=lambda x: int(x.name.split('_')[1]) if x.name.split('_')[1].isdigit() else 0)

    for f in fold_files:
        dataset_disponibili.append(f.name.replace("_train.csv", ""))

    # Cerca il dataset intero
    if (preprocessed_dir / "processed_full_tree.csv").exists():
        dataset_disponibili.append("processed_full_tree")

    if not dataset_disponibili:
        print("Errore: Nessun dataset per gli alberi trovato nella cartella data/preprocessed_folds.")
        return

    # =========================================================
    # 2. COSTRUZIONE DEL MENU INTERATTIVO
    # =========================================================
    mappa_dataset = {}
    print("\nScegli quale dataset utilizzare inserendo il numero corrispondente:")

    for i, nome in enumerate(dataset_disponibili, start=1):
        mappa_dataset[str(i)] = nome

        # Formattazione per rendere il menu leggibile
        if "holdout" in nome:
            desc = "Holdout (Train/Test split)"
        elif "kfold" in nome:
            numero_fold = nome.split('_')[1]
            desc = f"K-Fold numero {numero_fold}"
        elif "full" in nome:
            desc = "Intero Dataset (Submission finale)"
        else:
            desc = nome

        print(f"{i}: {desc}")

    scelta = input(f"\nInserisci un numero da 1 a {len(dataset_disponibili)}: ").strip()

    while scelta not in mappa_dataset:
        print("Scelta non valida. Riprova.")
        scelta = input(f"Inserisci un numero da 1 a {len(dataset_disponibili)}: ").strip()

    dataset_scelto = mappa_dataset[scelta]

    # Generazione dei percorsi sicuri
    if dataset_scelto == 'processed_full_tree':
        train_path = preprocessed_dir / f"{dataset_scelto}.csv"
    else:
        train_path = preprocessed_dir / f"{dataset_scelto}_train.csv"

    test_path = preprocessed_dir / f"{dataset_scelto}_test.csv"

    outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"INIZIO ELABORAZIONE SINGOLA: {dataset_scelto.upper()}")
    print(f"{'=' * 60}")

    # ---------------------------------------------------------
    # 3. CARICAMENTO E CONVERSIONE DEI DATI
    # ---------------------------------------------------------
    print("[1/4] Caricamento in corso...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=['Transported', 'PassengerId'], errors='ignore')
    y_train = train_df['Transported']
    X_test = test_df.drop(columns=['Transported', 'PassengerId'], errors='ignore')

    # Convertiamo tutte le colonne testuali in categorie per LightGBM
    colonne_testuali = X_train.select_dtypes(include=['object']).columns
    for col in colonne_testuali:
        X_train[col] = X_train[col].astype('category')
        if col in X_test.columns:
            X_test[col] = X_test[col].astype('category')

    if 'PassengerId' in test_df.columns:
        passenger_ids = test_df['PassengerId']
    else:
        passenger_ids = [f"ID_Fold_{i}" for i in range(len(test_df))]

    # ---------------------------------------------------------
    # 4. INIZIALIZZAZIONE DEL MODELLO
    # ---------------------------------------------------------
    print("[2/4] Inizializzazione del modello LightGBM...")
    trainer = LightGBMTrainer(random_state=42)

    # ---------------------------------------------------------
    # 5. ADDESTRAMENTO E TUNING
    # ---------------------------------------------------------
    print("[3/4] Ottimizzazione e Addestramento in corso...")
    trainer.tune_hyperparameters(X_train, y_train)

    # ---------------------------------------------------------
    # 6. PREDIZIONE E SALVATAGGIO
    # ---------------------------------------------------------
    print("[4/4] Generazione predizioni e salvataggio file...")
    predictions = trainer.predict(X_test)

    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Transported': predictions.astype(bool)
    })

    submission_filename = outputs_dir / f"submission_lightgbm_{dataset_scelto}.csv"
    submission.to_csv(submission_filename, index=False)

    model_filename = outputs_dir / f"modello_lightgbm_{dataset_scelto}.pkl"
    joblib.dump(trainer.best_model, model_filename)

    print(f"Previsioni salvate in: {submission_filename}")
    print(f"Modello salvato in: {model_filename}")
    print("\nELABORAZIONE COMPLETATA CON SUCCESSO!")


if __name__ == "__main__":
    main()