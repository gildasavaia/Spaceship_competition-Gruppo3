import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class SVCTrainer:
    """ Classe per gestire l'addestramento, l'ottimizzazione e le predizioni
    del modello Support Vector Classifier (SVC) per lo Spaceship Titanic."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_model = None

    def tune_hyperparameters(self, X, y, param_grid=None, cv_splits=5, fast_mode=True):
        """
        Ottimizza gli iperparametri dell'SVC usando RandomizedSearchCV o utilizza parametri ottimali predefiniti.

        Args:
            X: Features di training.
            y: Target di training.
            param_grid: Griglia di parametri per la ricerca (se fast_mode=False).
            cv_splits: Numero di fold per la Cross Validation (se fast_mode=False).
            fast_mode: Se True, usa i parametri ideali "scolpiti nel codice". Se False, avvia la ricerca.
        """

        # È FONDAMENTALE scalare i dati per l'SVC, quindi creiamo una Pipeline che
        # faccia lo scaling prima di passare i dati al modello.
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(probability=True, random_state=self.random_state))
        ])

        if fast_mode:
            print("🚀 [Fast Mode] Utilizzo parametri SVC ottimali pre-impostati (Nessuna ricerca!)...")

            # Parametri ottimali di default.
            # (Nota: L'SVC con kernel RBF è spesso il più potente)
            pipeline.set_params(
                svc__C=10.0,
                svc__kernel='rbf',
                svc__gamma='scale'
            )

            self.best_model = pipeline
            self.best_model.fit(X, y)
            print("✅ Addestramento SVC completato!")
            return self.best_model

        else:
            print("🔍 Inizio ottimizzazione intelligente (RandomizedSearchCV) per SVC...")

            if param_grid is None:
                # Griglia di ricerca per RandomizedSearch
                param_grid = {
                    'svc__C': [0.1, 1, 10, 100],
                    'svc__gamma': ['scale', 'auto', 0.1, 0.01],
                    'svc__kernel': ['rbf', 'linear']
                }

            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

            # Usiamo RandomizedSearchCV. n_iter=10 prova 10 combinazioni casuali.
            random_search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_grid,
                n_iter=10,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,  # Usa tutti i core disponibili
                verbose=1,
                random_state=self.random_state
            )

            random_search.fit(X, y)
            self.best_model = random_search.best_estimator_

            print(f"✅ Migliori parametri trovati: {random_search.best_params_}")
            print(f"Miglior accuratezza in CV con tuning: {random_search.best_score_:.4f}")

            return self.best_model

    def predict(self, X_test):
        """Effettua le predizioni sui nuovi dati."""
        if self.best_model is None:
            raise Exception("Devi prima addestrare il modello chiamando tune_hyperparameters()!")
        return self.best_model.predict(X_test)

    def predict_proba(self, X):
        """Restituisce le probabilità della classe positiva (Transported=True)."""
        if self.best_model is None:
            raise Exception("Il modello deve essere addestrato prima.")
        return self.best_model.predict_proba(X)[:, 1]