import pandas as pd
import numpy as np
import joblib
import glob
import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from Evaluation.metrics_calculator import MetricsEvaluator
from Random_Forest_Classifier_model import RandomForestTrainer





def esegui_pipeline_rf(train_path, test_path, dataset_name, outputs_dir, salva_file_singolo=True):
    """Questa funzione esegue l'intera pipeline Random_Forest_Classifier per una specifica coppia di Train/Test.
    Carica i CSV, prepara i dati per Random_Forest_Classifier, addestra il modello, valuta le metriche e salva gli output."""

    print(f"\n{'-' * 60}")
    print(f"INIZIO ELABORAZIONE Random Forest Classifier: {dataset_name.upper()}")
    print(f"{'-' * 60}")

    # 1) Caricamento dai dati attraverso la lettura dei file CSV specificati nei percorsi.
    print("[1/4] Caricamento dati in corso...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Separazione delle feature (X) dal target (y), rimuovendo le colonne non predittive.
    X_train = train_df.drop(columns=['Transported', 'PassengerId'], errors='ignore')
    y_train = train_df['Transported']
    X_test = test_df.drop(columns=['Transported', 'PassengerId'], errors='ignore')

    # Uniamo verticalmente Train e Test prima dell'Encoding.
    # In questo modo garantiamo che vengano generate le stesse identiche Dummy variables per entrambi.
    # Salviamo il numero di righe del Train per poter dividere correttamente il dataset.
    num_train_rows = X_train.shape[0]

    # Uniamo verticalmente Train e Test.
    combined_df = pd.concat([X_train, X_test], axis=0, ignore_index=True)

    # Identificazione delle colonne caratterizzate da stringhe, convertendole in tipo category.
    colonne_testuali = combined_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if colonne_testuali:
        print(f"*** Trasformazione variabili categoriche: {colonne_testuali} ***")
        # One-Hot Encoding (crea colonne binarie 0/1)
        combined_df = pd.get_dummies(combined_df, columns=colonne_testuali, drop_first=True)

    # Forza tutto il dataset a essere numerico (float).
    combined_df = combined_df.astype(float)

    # Dividiamo nuovamente i due dataset, ora perfettamente allineati e numerici.
    X_train = combined_df.iloc[:num_train_rows, :].copy()
    X_test = combined_df.iloc[num_train_rows:, :].copy()

    # Salviamo i PassengerId per poterli inserire nel file di submission finale.
    passenger_ids = test_df['PassengerId'] if 'PassengerId' in test_df.columns else range(len(test_df))

    # 2) Addestramento e ottimizzazione, istanziamo il nostro Trainer e lo avviamo.
    print("[2/4] Ricerca iperparametri ottimali (GridSearch)...")
    trainer = RandomForestTrainer(random_state=42)
    trainer.tune_hyperparameters(X_train, y_train)

    # 3) Chiediamo al modello di prevedere sia la classe finale (0/1) sia la probabilità continua [0,1].
    print("[3/4] Generazione predizioni e probabilità per Stacking...")
    predictions = trainer.predict(X_test) # Array di True/False.
    probabilities = trainer.predict_proba(X_test) # Array di probabilità decimali (0.0 - 1.0).

    # Valutazione delle metriche.
    if 'Transported' in test_df.columns:
        valutatore = MetricsEvaluator(
            y_true=test_df['Transported'],
            y_pred=predictions,
            y_probs=probabilities,
            dataset_name="Random Forest (Holdout)"
        )

        # Stampa le metriche solo se stiamo facendo l'Holdout o salvando file singoli.
        if salva_file_singolo:
            valutatore.print_report()
            valutatore.plot_visuals()

    # Creazione dei Dataframe di output in cui inserire i risultati per l'esportazione in CSV.
    res_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Transported': predictions.astype(bool)
    })
    prob_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Probability': probabilities
    })

    # 4) Salvataggio del modello per poterlo riutilizzare in futuro senza riaddestrare.
    print("[4/4] Salvataggio risultati in 'outputs'...")
    outputs_dir.mkdir(parents=True, exist_ok=True)
   

    # Salvataggio dei singoli file CSV, sia la predizione della classe e sia la probabilità.
    if salva_file_singolo:
        sub_file = outputs_dir / f"submission_rf_{dataset_name}.csv"
        res_df.to_csv(sub_file, index=False)


    # Catturiamo le risposte vere direttamente dal dataset di test del fold per usarle nell'aggregazione del K-Fold.
    y_test_true = test_df['Transported'] if 'Transported' in test_df.columns else None

    return res_df, prob_df, y_test_true, predictions, probabilities


def main():
    print("Avvio della pipeline Random Forest per Spaceship Titanic...\n")

    # Configurazione dinamica dei percorsi per far girare lo script su qualsiasi macchina.
    base_dir = Path(__file__).resolve().parent.parent
    preprocessed_dir = base_dir / "data" / "preprocessed_folds"
    outputs_dir = base_dir / "outputs"

    # Verifica che il preprocessing sia stato completato.
    if not preprocessed_dir.exists():
        print(f"Errore: Assicurati che i file 'full_tree' esistano in {preprocessed_dir}")
        return

    print("Seleziona il metodo di addestramento per Random Forest:")
    print("1. Holdout (singolo train/test)")
    print("2. K-Fold (addestramento automatico e output UNIFICATO in file TOTAL)")
    print("3. Processed Full (preparazione della submission finale per Kaggle)")

    scelta = input("Inserisci 1, 2 o 3: ").strip()

    # Opzione 1: Holdout Validation.
    if scelta == "1":
        train_path = preprocessed_dir / "holdout_tree_train.csv"
        test_path = preprocessed_dir / "holdout_tree_test.csv"

        if train_path.exists() and test_path.exists():
            esegui_pipeline_rf(train_path, test_path, "holdout_tree", outputs_dir, salva_file_singolo=True)
        else:
            print(f"Errore: Assicurati che i file 'holdout_tree' esistano in {preprocessed_dir}")

    # Opzione 2: K-Fold Cross Validation.
    elif scelta == "2":

        print("\nRicerca dei file K-Fold in corso...")
        search_pattern = str(preprocessed_dir / "kfold_*_tree_train.csv")
        train_files = glob.glob(search_pattern)

        if not train_files:
            print(f"Errore: Assicurati che i file 'kfold_*_tree_train.csv' esistano in {preprocessed_dir}")
        else:
            num_folds = len(train_files)
            print(f"Trovati {num_folds} fold. Avvio elaborazione in serie...\n")

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
                    print(f"⚠️ File test mancante per il fold {i}. Salto...")
                    continue

                # Eseguiamo la pipeline senza salvare i singoli file parziali per singolo fold in modo da non sporcare la cartella.
                res, prob, y_true, preds, probs = esegui_pipeline_rf(
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
            print("\n[*] Unione di tutte le predizioni K-Fold in un unico file TOTAL...")
            # Unisce e ordina per PassengerId
            final_res = pd.concat(all_res).sort_values('PassengerId')
            final_prob = pd.concat(all_prob).sort_values('PassengerId')

            final_res.to_csv(outputs_dir / "submission_rf_kfold_TOTAL.csv", index=False)
            final_prob.to_csv(outputs_dir / "prob_rf_kfold_TOTAL.csv", index=False)
            print(f"Creato UNICO file di submission per Random Forest!")

            # Calcoliamo una metrica globale aggregando le risposte di tutti i 5 fold fusi insieme.
            print("\nCalcolo delle metriche globali sull'intero K-Fold...")
            if all_y_true:
                valutatore = MetricsEvaluator(
                    y_true=np.array(all_y_true),
                    y_pred=np.array(all_y_pred),
                    y_probs=np.array(all_y_probs),
                    dataset_name="Random Forest (K-Fold)"
                )
                valutatore.print_report()
                valutatore.plot_visuals()
            else:
                print("Dati veri mancanti. Impossibile calcolare le metriche globali.")

    # Opzione 3: Full Validation.
    elif scelta == "3":
        train_path = preprocessed_dir / "processed_full_tree.csv"
        test_path = preprocessed_dir / "processed_full_tree_test.csv"

        if train_path.exists() and test_path.exists():
            esegui_pipeline_rf(train_path, test_path, "processed_full_tree", outputs_dir, salva_file_singolo=True)
        else:
            print(f"Errore: Assicurati che i file 'full_tree' esistano in {preprocessed_dir}")

    else:
        print("Scelta non valida. Riavvia lo script.")


if __name__ == "__main__":
    main()