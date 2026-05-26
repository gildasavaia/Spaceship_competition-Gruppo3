from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class SVCTrainer:
    """ Questa classe gestisce l'addestramento, l'ottimizzazione e le predizioni del modello Support Vector Classifier."""

    def __init__(self, random_state=42):

        # Inizializziamo il random state per garantire la riproducibilità dei risultati.
        self.random_state = random_state

        # Variabile per memorizzare il modello addestrato.
        self.best_model = None


    def tune_hyperparameters(self, X, y, param_grid=None, cv_splits=5, fast_mode=True):
        """ Questa funzione gestisce l'addestramento e l'ottimizzazione degli iperparametri del modello.
                Vediamo la logica concettuale:

                - FAST MODE: Si utilizza un set di iperparametri pre-impostati direttamente nel codice che sono stati
                 individuati come ottimali, garantendo un addestramento quasi istantaneo.

                - TUNING MODE: Modalità esplorativa, ad esempio tramite RandomizedSearchCV o GridSearchCV, in cui l'algoritmo
                 testa svariate o tutte le combinazioni di parametri per scovare la migliore.

                STATO ATTUALE DELL'IMPLEMENTAZIONE: Per ragioni di performance e per velocizzare l'esecuzione della pipeline,
                abbiamo implementato esclusivamente la logica Fast Mode. Attraverso aggiornamenti del codice, sarà possibile
                integrare le librerie di Search per abilitare la logica Tuning Mode.

                Argomenti in input:
                X: Features di training.
                y: Target di training.
                param_grid: Griglia di parametri per la ricerca, se fast_mode=False.
                cv_splits: Numero di fold per la Cross Validation, se fast_mode=False.
                fast_mode: Se True, usa i parametri preimpostati. Se False, avvia la ricerca."""

        # Creazione di una Pipeline che effettui lo scaling prima di passare i dati al modello.
        # Questo previene il Data Leakage perché lo scaler viene fittato solo sui dati di Train.
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(probability=True, random_state=self.random_state))
        ])

        if fast_mode:
            print("[Fast Mode] Utilizzo iperparametri ottimali pre-impostati...")

            # Parametri ottimali di default.
            pipeline.set_params(
                svc__C=10.0,           # Penalità per gli errori, più è alto e più cerca di non sbagliare.
                svc__kernel='rbf',     # Permette di creare confini decisionali curvi e complessi.
                svc__gamma='scale'     # Definisce il raggio d'influenza dei singoli punti di addestramento.
            )

            # Passiamo al modello la migliore combinazione di iperparametri trovati.
            self.best_model = pipeline

            # Addestramento del modello sui dati passati in input (X e y).
            self.best_model.fit(X, y)

            print("Addestramento completato con successo.")
            return self.best_model

        else:
            print("[Tuning Mode] Inizio ottimizzazione con RandomizedSearchCV...")

            if param_grid is None:
                # Se l'utente non ha passato una griglia personalizzata, usiamo questa di default.
                param_grid = {
                    'svc__C': [0.1, 1, 10, 100],
                    'svc__gamma': ['scale', 'auto', 0.1, 0.01],
                    'svc__kernel': ['rbf', 'linear']
                }

            # Configuriamo la Cross-Validation per valutare il tuning.
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

            # Configuriamo la griglia di ricerca randomizzata RandomizedSearchCV.
            random_search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_grid,
                n_iter=10,                            # Sceglie 10 combinazioni a caso dalla griglia.
                cv=cv,
                scoring='accuracy',                   # Valuta in base all'accuratezza delle risposte.
                n_jobs=-1,
                verbose=1,                            # Mostra i log durante il caricamento.
                random_state=self.random_state        # Seed per garantire che la randomizzazione sia fissa.
            )

            # Avvio del processo di ricerca e addestramento sulle varie combinazioni.
            random_search.fit(X, y)

            # Salvataggio nella variabile del miglior modello emerso dalla ricerca.
            self.best_model = random_search.best_estimator_

            print(f"Migliori parametri trovati: {random_search.best_params_}")
            print(f"Miglior accuratezza in CV con tuning: {random_search.best_score_:.4f}")

            return self.best_model


    def predict(self, X_test):
        """ Questa funzione effettua le predizioni binarie sui nuovi dati."""

        # Impediamo di fare predizioni se il modello non è stato prima addestrato.
        if self.best_model is None:
            raise Exception("Nessun modello addestrato trovato.")

        # Restituisce l'array delle classi predette.
        return self.best_model.predict(X_test)


    def predict_proba(self, X):
        """ Restituisce le probabilità della classe positiva."""

        # Impediamo di fare predizioni se il modello non è stato prima addestrato.
        if self.best_model is None:
            raise Exception("Nessun modello addestrato trovato.")

        # predict_proba restituisce due colonne, ovvero la probabilità della classe 0 e della classe 1 per ogni campione.
        # Con [:, 1] diciamo di estrarre solo la seconda colonna, ovvero la probabilità che il passeggero sia Trasportato
        return self.best_model.predict_proba(X)[:, 1]