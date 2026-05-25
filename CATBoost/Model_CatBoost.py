import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_data(train_path, test_path):
    """
    Esegue la lettura dei file CSV di addestramento e di test tramite pandas.

    Restituisce due DataFrame distinti (train, test).
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def prepare_data(train_df):
    """
    Isola la variabile target 'Transported' dalle feature del modello.

    Restituisce il DataFrame X (le feature) e la Series y (i target).
    """
    X = train_df.drop("Transported", axis=1)
    y = train_df["Transported"]
    return X, y


def prepare_test(test_df):
    """
    Crea una copia indipendente del DataFrame di test per prevenire
    problemi di SettingWithCopyWarning durante le elaborazioni successive.
    """
    return test_df.copy()


def create_catboost_model():
    """
    Inizializza e configura un'istanza di CatBoostClassifier impostando
    gli iperparametri ottimali per la regolarizzazione e l'addestramento.
    """
    return CatBoostClassifier(
        iterations=1400,  # Numero massimo di alberi da costruire
        learning_rate=0.035,  # Passo di aggiornamento dei pesi del gradiente
        depth=5,  # Profondità massima degli alberi decisionali
        l2_leaf_reg=7,  # Coefficiente di regolarizzazione L2 sulle foglie
        random_strength=0.8,  # Quantità di rumore da aggiungere nello split dei nodi
        bootstrap_type='Bernoulli',  # Metodo di campionamento delle righe
        subsample=0.8,  # Percentuale di righe campionate ad ogni iterazione
        rsm=0.85,  # Percentuale di feature campionate ad ogni split
        loss_function='Logloss',  # Funzione obiettivo per classificazione binaria
        eval_metric='Accuracy',  # Metrica utilizzata per monitorare l'overfitting
        od_type='Iter',  # Tipo di Overfitting Detector basato sulle iterazioni
        od_wait=80,  # Numero di iterazioni di tolleranza prima dell'early stopping
        thread_count=-1,  # Utilizzo di tutti i core di calcolo disponibili della CPU
        leaf_estimation_iterations=1,  # Numero di iterazioni per il calcolo dei valori delle foglie
        random_seed=42,  # Seed per garantire la riproducibilità dei risultati
        verbose=100  # Frequenza di stampa dei log di addestramento a schermo
    )


def train_model(model, X_train, y_train):
    """
    Rileva automaticamente le colonne di tipo stringa o object nel dataset,
    le classifica come feature categoriche e avvia il fit del modello.
    """
    cat_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    # Avvio del processo di ottimizzazione dei parametri
    model.fit(
        X_train, y_train,
        cat_features=cat_features,
        verbose=200
    )
    return model


def predict(model, X):
    """
    Genera predizioni binarie discrete (0 o 1) a partire dai dati di input.
    Incapsula i dati in una struttura CatBoost Pool per preservare la coerenza delle feature categoriche.
    """
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()
    test_pool = Pool(data=X, cat_features=cat_features)

    # Estrazione delle probabilità associate alla classe positiva ed applicazione della soglia decisionale a 0.5
    probs = model.predict_proba(test_pool)[:, 1]
    return (probs > 0.5).astype(int)


def save_submission(model, X_test, original_test_df, filename="submission_catboost.csv"):
    """
    Applica il modello addestrato sul set di test, mappa le predizioni in booleani
    e salva il file finale formattato secondo le specifiche di Kaggle.
    """
    # Rimuove la colonna PassengerId prima di passare i dati al modello, poiché non costituisce una feature predittiva
    X_input = X_test.drop("PassengerId", axis=1) if "PassengerId" in X_test.columns else X_test

    # Generazione dei valori predetti in formato intero
    preds = predict(model, X_input)

    # Costruzione del file di sottomissione finale con conversione del target in booleano (True/False)
    submission = pd.DataFrame({
        "PassengerId": original_test_df["PassengerId"],
        "Transported": preds.astype(bool)
    })

    # Scrittura del file CSV senza includere l'indice numerico di riga di pandas
    submission.to_csv(filename, index=False)
    print(f"\n Submission salvata in: {filename}")