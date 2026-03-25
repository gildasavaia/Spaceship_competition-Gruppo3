import pandas as pd
from LightGBM_model import LightGBMTrainer


def main():
    print("🚀 Avvio della pipeline di Machine Learning per Spaceship Titanic...")

    # 1. CARICAMENTO DEI DATI (Dal Feature Engineer)
    print("\n[1/5] Caricamento dei dati pre-processati...")
    try:
        train_df = pd.read_csv("train_processed_example.csv")
        test_df = pd.read_csv("test_processed_example.csv")
    except FileNotFoundError:
        print("Errore: Impossibile trovare i file CSV. Assicurati che il Feature Engineer li abbia generati!")
        return

    # Prepariamo la matrice delle feature (X) e il target (y).
    # Rimuoviamo il PassengerId perché non è una variabile utile per l'addestramento.
    X_train = train_df.drop(columns=['Transported', 'PassengerId'])
    y_train = train_df['Transported']

    # Per i dati di test, conserviamo gli ID in una variabile a parte per la submission finale.
    X_test = test_df.drop(columns=['PassengerId'])
    passenger_ids = test_df['PassengerId']


    # 2. INIZIALIZZAZIONE DEL MODELLO
    print("\n[2/5] Inizializzazione del modello LightGBM...")
    trainer = LightGBMTrainer(random_state=42)

    # (Opzionale) Valutazione di base per vedere come si comporta senza modifiche.
    trainer.evaluate_baseline(X_train, y_train)

    # 3. ADDESTRAMENTO E TUNING
    print("\n[3/5] Ottimizzazione e Addestramento in corso...")
    # NOTA: Se per fare dei test veloci non vuoi aspettare la GridSearch,
    # commenta la riga qui sotto e scommenta "trainer.train(X_train, y_train)".

    trainer.tune_hyperparameters(X_train, y_train)
    # trainer.train(X_train, y_train)

    # 4. ANALISI E PREDIZIONE
    print("\n[4/5] Generazione del grafico dell'importanza delle feature...")
    # Questo aprirà una finestra con il grafico. Chiudila per far proseguire lo script.
    trainer.show_feature_importance()

    print("\n[5/5] Generazione delle predizioni per i passeggeri smarriti...")
    predictions = trainer.predict(X_test)

    # 5. SALVATAGGIO DEI RISULTATI (Submission)
    # Formattiamo l'output per combaciare esattamente con sample_submission.csv
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Transported': predictions.astype(bool)  # Kaggle richiede True/False
    })

    submission_filename = "submission_lightgbm.csv"
    submission.to_csv(submission_filename, index=False)
    print(f"\n✅ Finito! Il file '{submission_filename}' è stato salvato con successo.")

if __name__ == "__main__":
    main()