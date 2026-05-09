import pandas as pd
import os
import joblib
from pathlib import Path
from Support_Vector_Classifier_model import SupportVectorTrainer


def main():
    print("Avvio della pipeline Support Vector Classifier per Spaceship Titanic...")

    # Gestione dinamica dei percorsi
    base_dir = Path(__file__).resolve().parent.parent
    preprocessed_dir = base_dir / "data" / "preprocessed_folds"
    outputs_dir = base_dir / "outputs"

    if not preprocessed_dir.exists():
        print(f"Errore: La cartella {preprocessed_dir} non esiste. Esegui prima la pipeline!")
        return

    # =========================================================
    # 2. RICERCA DINAMICA DEI DATASET
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
    # 3. MENU INTERATTIVO
    # =========================================================
    mappa_dataset = {}
    print("\nScegli il dataset per SVC:")

    for i, nome in enumerate(dataset_disponibili, start=1):
        mappa_dataset[str(i)] = nome
        print(f"{i}: {nome}")

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
    print(f"ELABORAZIONE SVC: {dataset_scelto.upper()}")
    print(f"{'=' * 60}")

    # ---------------------------------------------------------
    # 4. CARICAMENTO E PREPARAZIONE DATI
    # ---------------------------------------------------------
    print("[1/4] Caricamento dati in corso...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=['Transported', 'PassengerId'], errors='ignore')
    y_train = train_df['Transported']
    X_test = test_df.drop(columns=['Transported', 'PassengerId'], errors='ignore')

    # =====================================================================
    # TRUCCO ROBUSTO: Unisci, Converti e Dividi (One-Hot Encoding)
    # L'SVC richiede dati puramente numerici. Trasformiamo le stringhe.
    # =====================================================================
    num_train_rows = X_train.shape[0]

    # Uniamo temporaneamente per avere lo stesso numero di colonne
    combined_df = pd.concat([X_train, X_test], axis=0, ignore_index=True)

    # Identifichiamo le colonne testuali
    colonne_testuali = combined_df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    if colonne_testuali:
        print(f"[*] Conversione variabili categoriche: {colonne_testuali}")
        combined_df = pd.get_dummies(combined_df, columns=colonne_testuali, drop_first=True)

    # Forza la conversione di tutto in float (decimale) per lo StandardScaler
    combined_df = combined_df.astype(float)

    # Dividiamo nuovamente i dataset
    X_train = combined_df.iloc[:num_train_rows, :].copy()
    X_test = combined_df.iloc[num_train_rows:, :].copy()
    # =====================================================================

    passenger_ids = test_df['PassengerId'] if 'PassengerId' in test_df.columns else range(len(test_df))

    # ---------------------------------------------------------
    # 5. ADDESTRAMENTO E TUNING
    # ---------------------------------------------------------
    print("[2/4] Ottimizzazione iperparametri SVC (GridSearch)...")
    trainer = SupportVectorTrainer(random_state=42)
    trainer.tune_hyperparameters(X_train, y_train)

    # ---------------------------------------------------------
    # 6. PREDIZIONI E PROBABILITÀ
    # ---------------------------------------------------------
    print("[3/4] Generazione predizioni e probabilità per lo Stacking...")
    predictions = trainer.predict(X_test)
    probabilities = trainer.predict_proba(X_test)

    # ---------------------------------------------------------
    # 7. SALVATAGGIO OUTPUT
    # ---------------------------------------------------------
    print("[4/4] Salvataggio risultati in 'outputs'...")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Submission Kaggle (Vero/Falso)
    sub_file = outputs_dir / f"submission_svc_{dataset_scelto}.csv"
    pd.DataFrame({
        'PassengerId': passenger_ids,
        'Transported': predictions.astype(bool)
    }).to_csv(sub_file, index=False)

    # 2. Probabilità Stacking (Numeri tra 0 e 1)
    prob_file = outputs_dir / f"prob_svc_{dataset_scelto}.csv"
    pd.DataFrame({
        'PassengerId': passenger_ids,
        'Probability': probabilities
    }).to_csv(prob_file, index=False)

    # 3. Salvataggio Modello (.pkl)
    model_file = outputs_dir / f"modello_svc_{dataset_scelto}.pkl"
    joblib.dump(trainer.best_model, model_file)

    print(f"\n✅ Pipeline SVC completata!")
    print(f"-> File per Kaggle: {sub_file.name}")
    print(f"-> File per Stacking: {prob_file.name}")
    print(f"-> Modello salvato: {model_file.name}")


if __name__ == "__main__":
    main()