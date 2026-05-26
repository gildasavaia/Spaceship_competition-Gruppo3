import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

class LightGBMTrainer:
    """ Questa classe gestisce l'addestramento, l'ottimizzazione e le predizioni del modello LightGBM."""

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
        integrare le librerie di Search per abilitare la logica Tuning Mode."""

        if fast_mode:
            print("[Fast Mode] Utilizzo iperparametri ottimali pre-impostati...")
            # Passiamo al modello la migliore combinazione di iperparametri trovati.
            self.best_model = lgb.LGBMClassifier(
                num_leaves=31,                        # Numero massimo di foglie in un albero.
                max_depth=7,                          # Profondità massima dell'albero per evitare l'overfitting.
                learning_rate=0.05,                   # Passo di apprendimento, è lento ma più preciso.
                n_estimators=300,                     # Numero di alberi decisionali da costruire.
                random_state=self.random_state,       # Fissa il seed per l'aleatorietà
                n_jobs=-1                             # Usa tutti i core disponibili della CPU per velocizzare.
            )

            # Addestramento del modello sui dati passati in input (X e y).
            self.best_model.fit(X, y)
            print("Addestramento completato con successo.")
            return self.best_model


        else:
            print("[Tuning Mode] Inizio ottimizzazione con RandomizedSearchCV...")
            # Inizializziamo un modello "vuoto" di base.
            base_model = lgb.LGBMClassifier(random_state=self.random_state, n_jobs=-1)

            # Se l'utente non ha passato una griglia personalizzata, usiamo questa di default.
            if param_grid is None:
                param_grid = {
                    'num_leaves': [20, 31, 50, 70],
                    'max_depth': [5, 7, -1],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [100, 300, 500]
                }

            # Configuriamo la Cross-Validation per valutare il tuning.
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

            # Configuriamo la griglia di ricerca randomizzata RandomizedSearchCV.
            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=10,                           # Numero di combinazioni casuali da testare, ne abbiamo scelte 10 ma possono essere maggiori.
                cv=cv,                               # La strategia di validazione.
                scoring='accuracy',                  # Metrica di valutazione da massimizzare.
                n_jobs=-1,
                verbose=1,                           # Mostra i log durante il caricamento.
                random_state=self.random_state       # Seed per garantire che la randomizzazione sia fissa.
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