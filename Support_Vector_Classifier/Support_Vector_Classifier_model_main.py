import pandas as pd
import os
import joblib
import warnings
from pathlib import Path
from Support_Vector_Classifier_model import SupportVectorTrainer

# Ignoriamo i warning per un terminale pulito
warnings.filterwarnings('ignore')


def main():
    print("Avvio della pipeline Support Vector Classifier per Spaceship Titanic...")

    base_dir = Path(__file__).resolve().parent.parent
    preprocessed_dir = base_dir / "data" / "preprocessed_folds"
    outputs_dir = base_dir / "outputs"

    if not preprocessed_dir.exists():
        print(f"Errore: Cartella {preprocessed_dir} non trovata.")
        return

    # =========================================================
    # 1. RICERCA DINAMICA DEI DATASET
    # =========================================================
    dataset_disponibili = []
    # Cerchiamo i file generati (Holdout, K-Fold, Full)
    if (preprocessed_dir / "holdout_tree_train.csv").exists():
        dataset_disponibili.append("holdout_tree")

    fold_files = list(preprocessed_dir.glob("kfold_*_tree_train.csv"))
    fold_files.sort(key=lambda x: int(x.name.split('_')[1]) if x.name.split('_')[1].isdigit() else 0)
    for f in fold_files:
        dataset_disponibili.append(f.name.replace("_train.csv", ""))

    if (preprocessed_dir / "processed_full_tree.csv").exists():
        dataset_disponibili.append("processed_full_tree")

    # =========================================================
    # 2. MENU INTERATTIVO
    # =========================================================
    mappa_dataset = {}
    print("\nSeleziona il dataset per SVC:")
    for i, nome in enumerate(dataset_disponibili, start=1):
        mappa_dataset[str(i)] = nome
        print(f"{i}: {nome}")

    scelta = input(f"\nInserisci numero (1-{len(dataset_disponibili)}): ").strip()
    while scelta not in mappa_dataset:
        scelta = input("Riprova: ").strip()

    dataset_scelto = mappa_dataset[scelta]

    # Caricamento percorsi
    if dataset_scelto == 'processed_full_tree':
        train_path = preprocessed_dir / f"{dataset_scelto}.csv"
    else:
        train_path = preprocessed_dir / f"{dataset_scelto}_train.csv"
    test_path = preprocessed_dir / f"{dataset_scelto}_test.csv"

    # ---------------------------------------------------------
    # 3. CARICAMENTO E PREPARAZIONE DATI
    # ---------------------------------------------------------
    print("[1/4] Caricamento dati...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Nota: SVC non accetta testo. Ci affidiamo all'encoding della tua pipeline
    X_train = train_df.drop(columns=['Transported', 'PassengerId'], errors='ignore')
    y_train = train_df['Transported']
    X_test = test_df.drop(columns=['Transported', 'PassengerId'], errors='ignore')

    # SVC è LENTO su dataset grandi. Se scegli il dataset intero (8700 righe),
    # il PC potrebbe metterci qualche minuto per la Grid Search.

    if 'PassengerId' in test_df.columns:
        passenger_ids = test_df['PassengerId']
    else:
        passenger_ids = range(len(test_df))

    # ---------------------------------------------------------
    # 4. TRAINING E PREDIZIONE
    # ---------------------------------------------------------
    trainer = SupportVectorTrainer(random_state=42)

    print("[2/4] Ottimizzazione iperparametri...")
    trainer.tune_hyperparameters(X_train, y_train)

    print("[3/4] Generazione predizioni...")
    predictions = trainer.predict(X_test)

    # ---------------------------------------------------------
    # 5. SALVATAGGIO
    # ---------------------------------------------------------
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Transported': predictions.astype(bool)
    })

    sub_file = outputs_dir / f"submission_svc_{dataset_scelto}.csv"
    model_file = outputs_dir / f"modello_svc_{dataset_scelto}.pkl"

    outputs_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(sub_file, index=False)
    joblib.dump(trainer.best_model, model_file)

    print(f"\n✅ SVC completato con successo!")
    print(f"-> Modello salvato: {model_file}")


if __name__ == "__main__":
    main()