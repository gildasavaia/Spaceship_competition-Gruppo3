import pandas as pd
import numpy as np
import joblib
import warnings
import glob
import sys
from pathlib import Path

from Evaluation.metrics_calculator import MetricsEvaluator
from LightGBM_model import LightGBMTrainer

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

warnings.filterwarnings('ignore', category=UserWarning)


def esegui_pipeline_lightgbm(train_path, test_path, dataset_name, outputs_dir, salva_file_singolo=True):
    """Questa funzione esegue l'intera pipeline LightGBM per una specifica coppia di Train/Test. Carica i CSV, prepara i
    dati per LightGBM, addestra il modello, valuta le metriche e salva gli output."""

    print(f"\n{'-' * 60}")
    print(f"INIZIO ELABORAZIONE LightGBM: {dataset_name.upper()}")
    print(f"{'-' * 60}")

    # 1) Caricamento dai dati attraverso la lettura dei file CSV specificati nei percorsi.
    print("[1/4] Caricamento dati in corso...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Separazione delle feature (X) dal target (y), rimuovendo le colonne non predittive.
    X_train = train_df.drop(columns=['Transported', 'PassengerId'], errors='ignore')
    y_train = train_df['Transported']
    X_test = test_df.drop(columns=['Transported', 'PassengerId'], errors='ignore')

    # Il set di categorie sia identico tra Train e Test.
    num_train_rows = X_train.shape[0]

    # Uniamo verticalmente Train e Test.
    combined_df = pd.concat([X_train, X_test], axis=0, ignore_index=True)

    # Identificazione delle colonne caratterizzate da stringhe, convertendole in tipo category.
    colonne_testuali = combined_df.select_dtypes(include=['object', 'string']).columns.tolist()
    if colonne_testuali:
        print(f"*** Colonne testuali rilevate: {colonne_testuali}. Conversione in 'category'... ***\n")
        for col in colonne_testuali:
            combined_df[col] = combined_df[col].astype('category')

    # Ri-dividiamo il dataset unito tornando ai Train e Test originali, ma ora con i tipi corretti.
    X_train = combined_df.iloc[:num_train_rows, :].copy()
    X_test = combined_df.iloc[num_train_rows:, :].copy()

    # Salviamo i PassengerId per poterli inserire nel file di submission finale.
    passenger_ids = test_df['PassengerId'] if 'PassengerId' in test_df.columns else range(len(test_df))

    # 2) Addestramento e ottimizzazione, istanziamo il nostro Trainer e lo avviamo.
    print("[2/4] Inizializzazione e tuning del modello LightGBM...")
    trainer = LightGBMTrainer(random_state=42)
    trainer.tune_hyperparameters(X_train, y_train)

    # 3) Chiediamo al modello di prevedere sia la classe finale (0/1) sia la probabilità continua [0,1].
    print("\n[3/4] Generazione predizioni e probabilità per lo Stacking...")
    predictions = trainer.predict(X_test) # Array di True/False.
    probabilities = trainer.predict_proba(X_test) # Array di probabilità decimali (0.0 - 1.0).

    # Creazione dei Dataframe di output in cui inserire i risultati per l'esportazione in CSV.
    res_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Transported': predictions.astype(bool)
    })
    prob_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Probability': probabilities
    })

    # Valutazione delle metriche.
    if 'Transported' in test_df.columns:
        valutatore = MetricsEvaluator(
            y_true=test_df['Transported'],
            y_pred=predictions,
            y_probs=probabilities,
            dataset_name="LightGBM (Holdout)"
        )

        # Se non siamo in un ciclo K-Fold, stampiamo subito i grafici ed il report a schermo.
        if salva_file_singolo:
            valutatore.print_report()
            valutatore.plot_visuals()

    # 4) Salvataggio del modello per poterlo riutilizzare in futuro senza riaddestrare.
    print("\n[4/4] Salvataggio risultati nella cartella 'outputs'...")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    model_file = outputs_dir / f"modello_lightgbm_{dataset_name}.pkl"
    joblib.dump(trainer.best_model, model_file)

    # Salvataggio dei singoli file CSV, sia la predizione della classe e sia la probabilità.
    if salva_file_singolo:
        sub_file = outputs_dir / f"submission_lightgbm_{dataset_name}.csv"
        res_df.to_csv(sub_file, index=False)

        prob_file = outputs_dir / f"prob_lightgbm_{dataset_name}.csv"
        prob_df.to_csv(prob_file, index=False)
        print(f"File CSV salvati per {dataset_name}.")

    # Estrazione delle etichette reali per restituirle all'orchestratore nei calcoli successivi.
    y_test_true = test_df['Transported'] if 'Transported' in test_df.columns else None

    return res_df, prob_df, y_test_true, predictions, probabilities


def main():

    print("Avvio della pipeline LightGBM per Spaceship Titanic...\n")

    # Configurazione dinamica dei percorsi per far girare lo script su qualsiasi macchina.
    base_dir = Path(__file__).resolve().parent.parent
    preprocessed_dir = base_dir / "data" / "preprocessed_folds"
    outputs_dir = base_dir / "outputs"

    # Verifica che il preprocessing sia stato completato.
    if not preprocessed_dir.exists():
        print(f"Errore: La cartella {preprocessed_dir} non esiste. Eseguire la pipeline di preprocessing.")
        return

    print("Selezionare il metodo di addestramento per LightGBM:")
    print("1. Holdout Validation")
    print("2. K-Fold Cross Validation")
    print("3. Full Validation (preparazione della submission finale per competizione Kaggle)")

    scelta = input("Inserire il numero corrispondente: ").strip()

    # Opzione 1: Holdout Validation.
    if scelta == "1":
        train_path = preprocessed_dir / "holdout_tree_train.csv"
        test_path = preprocessed_dir / "holdout_tree_test.csv"

        if train_path.exists() and test_path.exists():
            esegui_pipeline_lightgbm(train_path, test_path, "holdout_tree", outputs_dir, salva_file_singolo=True)
        else:
            print("Errore: File holdout mancanti.")

    # Opzione 2: K-Fold Cross Validation.
    elif scelta == "2":
        print("\nRicerca dei file K-Fold in corso...")
        search_pattern = str(preprocessed_dir / "kfold_*_tree_train.csv")
        train_files = glob.glob(search_pattern)

        if not train_files:
            print("Errore: Nessun file K-Fold trovato nella cartella 'preprocessed_folds'")
        else:
            num_folds = len(train_files)
            print(f"Trovati {num_folds} fold. Avvio elaborazione...\n")

            # Inizializzazione liste vuote in cui inserire i risultati di tutti i fold.
            all_res = []
            all_prob = []
            all_y_true = []
            all_y_pred = []
            all_y_probs = []

            # Ciclo iterativo di addestramento su tutti i fold trovati.
            for i in range(1, num_folds + 1):
                train_path = preprocessed_dir / f"kfold_{i}_tree_train.csv"
                test_path = preprocessed_dir / f"kfold_{i}_tree_test.csv"

                if not test_path.exists():
                    print(f"File test mancante per il fold {i}.")
                    continue

                # Eseguiamo la pipeline senza salvare i singoli file parziali per singolo fold in modo da non sporcare la cartella.
                res, prob, y_true, preds, probs = esegui_pipeline_lightgbm(
                    train_path, test_path, f"kfold_{i}_tree", outputs_dir, salva_file_singolo=False
                )

                all_res.append(res)
                all_prob.append(prob)

                # Accodiamo le risposte e le previsioni nella lista.
                if y_true is not None:
                    all_y_true.extend(y_true)
                    all_y_pred.extend(preds)
                    all_y_probs.extend(probs)

            # Unione di tutte le predizioni K-Fold in un unico file TOTAL.
            final_res = pd.concat(all_res).sort_values('PassengerId')
            final_prob = pd.concat(all_prob).sort_values('PassengerId')

            final_res.to_csv(outputs_dir / "submission_lightgbm_kfold_TOTAL.csv", index=False)
            final_prob.to_csv(outputs_dir / "prob_lightgbm_kfold_TOTAL.csv", index=False)
            print(f"Creazione file CSV per k-fold.")

            # Calcoliamo una metrica globale aggregando le risposte di tutti i 5 fold fusi insieme.
            print("\nCalcolo delle metriche globali sull'intero K-Fold...")
            if all_y_true:
                valutatore = MetricsEvaluator(
                    y_true=np.array(all_y_true),
                    y_pred=np.array(all_y_pred),
                    y_probs=np.array(all_y_probs),
                    dataset_name="LIGHTGBM K-FOLD (GLOBALE)"
                )
                valutatore.print_report()
                valutatore.plot_visuals()
            else:
                print("Impossibile calcolare le metriche globali.")

    # Opzione 3: Full Validation.
    elif scelta == "3":
        # Usa il 100% dei dati di Train senza sprecarne una parte per la validazione
        train_path = preprocessed_dir / "processed_full_tree.csv"
        test_path = preprocessed_dir / "processed_full_tree_test.csv"

        if train_path.exists() and test_path.exists():
            esegui_pipeline_lightgbm(train_path, test_path, "processed_full_tree", outputs_dir, salva_file_singolo=True)
        else:
            print("Errore: File processed_full mancanti.")

    else:
        print("Scelta non valida. Riavvia lo script.")


if __name__ == "__main__":
    main()