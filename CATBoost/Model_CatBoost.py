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


"""
def tune_catboost_with_random_search(X, y, n_iter=20):
    
    MODALITÀ TUNING: Esegue la ricerca degli iperparametri tramite RandomizedSearchCV.

    A differenza di Optuna (che è probabilistico/Bayesiano), questo metodo estrae 
    in modo totalmente casuale 'n_iter' combinazioni all'interno dei range definiti.
    Esegue una Cross-Validation interna per ogni tentativo per evitare l'overfitting.
    
    print(f"\n--- Avvio TUNING MODE (RandomizedSearchCV) - {n_iter} tentativi ---")

    # Individuiamo le colonne categoriche, fondamentali per il corretto fit di CatBoost
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    # Inizializziamo il modello base.
    # Impostiamo verbose=0 per evitare che i log di ogni singolo albero intasino lo schermo.
    # Impostiamo thread_count=-1 per far viaggiare CatBoost alla massima velocità su tutti i core.
    base_model = CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='Accuracy',
        thread_count=-1,
        random_seed=42,
        verbose=0
    )

    # Definiamo lo spazio di ricerca (Griglia delle distribuzioni).
    # Usiamo liste per valori specifici e distribuzioni uniformi/intere per i range.
    param_distributions = {
        'iterations': [1000, 1500, 2000, 2500, 3000],  # Numero massimo di alberi
        'learning_rate': uniform(0.01, 0.04),  # Passo di aggiornamento (estratto tra 0.01 e 0.05)
        'depth': [4, 5, 6, 7, 8],  # Profondità massima degli alberi
        'l2_leaf_reg': randint(1, 10),  # Regolarizzazione L2 (valore intero casuale tra 1 e 9)
        'random_strength': uniform(0.5, 1.0),  # Rumore sugli split (tra 0.5 e 1.5)
        'subsample': [0.6, 0.7, 0.8, 0.9],  # Percentuale di righe campionate (Bernoulli)
        'bootstrap_type': ['Bernoulli']
    }

    # Configuriamos il modulo RandomizedSearchCV di scikit-learn
    # NOTA: Impostiamo n_jobs=1 o 2 se notiamo rallentamenti, perché CatBoost parallelizza 
    # già nativamente al suo interno sfruttando tutti i core (grazie a thread_count=-1).
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,  # Numero di combinazioni casuali che l'algoritmo testerà
        scoring='accuracy',  # La metrica di Kaggle che vogliamo massimizzare
        cv=5,  # 5-Fold Cross-Validation interna per ogni combinazione
        random_state=42,  # Garantisce la riproducibilità dell'estrazione casuale
        n_jobs=1,  # Lasciamo il parallelismo in mano al motore nativo di CatBoost
        verbose=1  # Mostra una barra di avanzamento sintetica dei test
    )

    print("Ricerca in corso sui fold della Cross-Validation...")
    # Avviamo la ricerca. Passiamo 'cat_features' all'interno del fit tramite kwargs
    random_search.fit(X, y, cat_features=cat_features)

    print("\n Tuning casuale completato!")
    print(f"Miglior accuratezza media stimata in CV: {random_search.best_score_:.4f}")
    print("Migliori iperparametri trovati:", random_search.best_params_)

    # Restituisce il dizionario con la combinazione vincente da dare in pasto al modello finale
    return random_search.best_params_
"""
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