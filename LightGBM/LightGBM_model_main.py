import pandas as pd
from LightGBM_model import LightGBMTrainer  # Importiamo la classe che hai creato!


def main():
    print("🚀 Avvio della pipeline di Machine Learning per Spaceship Titanic...")

    # ---------------------------------------------------------
    # 1. CARICAMENTO DEI DATI
    # ---------------------------------------------------------
    print("\n[1/4] Caricamento dei dati pre-processati...")
    try:
        train_df = pd.read_csv("../Dataset_Giocattolo/train_processed_example.csv")
        test_df = pd.read_csv("../Dataset_Giocattolo/test_processed_example.csv")
    except FileNotFoundError:
        print("Errore: Impossibile trovare i file CSV. Assicurati di averli nella stessa cartella!")
        return

    # Prepariamo la matrice delle feature (X) e il target (y)
    # Rimuoviamo il PassengerId perché non è una variabile utile per l'addestramento
    X_train = train_df.drop(columns=['Transported', 'PassengerId'])
    y_train = train_df['Transported']

    # Per i dati di test, conserviamo gli ID in una variabile a parte per la submission finale
    X_test = test_df.drop(columns=['PassengerId'])
    passenger_ids = test_df['PassengerId']

    # ---------------------------------------------------------
    # 2. INIZIALIZZAZIONE DEL MODELLO
    # ---------------------------------------------------------
    print("\n[2/4] Inizializzazione del modello LightGBM...")
    trainer = LightGBMTrainer(random_state=42)

    # (Opzionale) Valutazione di base per vedere come si comporta senza modifiche
    trainer.evaluate_baseline(X_train, y_train)

    # ---------------------------------------------------------
    # 3. ADDESTRAMENTO E TUNING
    # ---------------------------------------------------------
    print("\n[3/4] Ottimizzazione e Addestramento in corso...")

    trainer.tune_hyperparameters(X_train, y_train)
    # trainer.train(X_train, y_train) # Scommenta questa e commenta la riga sopra se vuoi saltare il tuning

    # ---------------------------------------------------------
    # 4. PREDIZIONE E SALVATAGGIO DEI RISULTATI (Submission)
    # ---------------------------------------------------------
    print("\n[4/4] Generazione delle predizioni per i passeggeri smarriti...")
    predictions = trainer.predict(X_test)

    # Formattiamo l'output per combaciare esattamente con sample_submission.csv
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Transported': predictions.astype(bool)  # Kaggle richiede True/False
    })

    submission_filename = "../outputs/submission_lightgbm.csv"
    submission.to_csv(submission_filename, index=False)

    # Stampa un piccolo resoconto per capire al volo come è andata
    conteggio = pd.Series(predictions).value_counts()
    percentuale_veri = conteggio.get(1, 0) / len(predictions) * 100
    print(f"\n--- RISULTATO PREDIZIONI ---")
    print(f"Passeggeri trasportati (True): {percentuale_veri:.1f}%")
    print(f"✅ Finito! Il file '{submission_filename}' è stato salvato con successo.")


if __name__ == "__main__":
    main()