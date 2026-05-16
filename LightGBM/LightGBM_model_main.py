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
    # 3. CARICAMENTO E PREPARAZIONE DATI
    # ---------------------------------------------------------
    print("[1/4] Caricamento dati in corso...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=['Transported', 'PassengerId'], errors='ignore')
    y_train = train_df['Transported']
    X_test = test_df.drop(columns=['Transported', 'PassengerId'], errors='ignore')

    # =====================================================================
    # TRUCCO SUPREMO PER LIGHTGBM: Unisci, Converti e Dividi
    # Garantisce che le categorie siano identiche al 100% tra Train e Test
    # =====================================================================
    num_train_rows = X_train.shape[0]

    # Uniamo temporaneamente train e test
    combined_df = pd.concat([X_train, X_test], axis=0, ignore_index=True)

    # Troviamo le colonne testuali nel calderone unito
    colonne_testuali = combined_df.select_dtypes(include=['object', 'string']).columns.tolist()

    if colonne_testuali:
        print(f"[*] Colonne testuali rilevate: {colonne_testuali}. Conversione in 'category'...")
        for col in colonne_testuali:
            combined_df[col] = combined_df[col].astype('category')

    # Ristacchiamo i dataset in modo perfetto
    X_train = combined_df.iloc[:num_train_rows, :].copy()
    X_test = combined_df.iloc[num_train_rows:, :].copy()
    # =====================================================================

    if 'PassengerId' in test_df.columns:
        passenger_ids = test_df['PassengerId']
    else:
        passenger_ids = range(len(test_df))

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
    # 6. PREDIZIONE E SALVATAGGIO (con Probabilità per Stacking)
    # ---------------------------------------------------------
    print("[4/4] Generazione predizioni, probabilità e salvataggio file...")

    # 1. Calcolo predizioni (True/False) e probabilità (0.0 - 1.0)
    predictions = trainer.predict(X_test)
    probabilities = trainer.predict_proba(X_test)

    # 2. Creazione dei DataFrame
    # Questo è per Kaggle
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Transported': predictions.astype(bool)
    })

    # Questo è per il nostro Meta-Modello (Ensemble)
    stacking_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Probability': probabilities
    })

    # 3. Definizione dei percorsi di salvataggio
    outputs_dir.mkdir(parents=True, exist_ok=True)  # Sicurezza extra
    submission_filename = outputs_dir / f"submission_lightgbm_{dataset_scelto}.csv"
    prob_filename = outputs_dir / f"prob_lightgbm_{dataset_scelto}.csv"
    model_filename = outputs_dir / f"modello_lightgbm_{dataset_scelto}.pkl"

    # 4. Salvataggio effettivo su disco
    submission.to_csv(submission_filename, index=False)
    stacking_df.to_csv(prob_filename, index=False)
    joblib.dump(trainer.best_model, model_filename)

    # 5. Messaggi finali
    print(f"Previsioni (Kaggle) salvate in: {submission_filename.name}")
    print(f"Probabilità (Stacking) salvate in: {prob_filename.name}")
    print(f"Modello (Cervello) salvato in: {model_filename.name}")
    print("\nELABORAZIONE COMPLETATA CON SUCCESSO!")


if __name__ == "__main__":
    main()