import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score


class LightGBMTrainer:

    """ Classe per gestire l'addestramento, l'ottimizzazione e le predizioni
    del modello LightGBM per lo Spaceship Titanic."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        # Inizializziamo il modello base.
        self.base_model = lgb.LGBMClassifier(
            random_state=self.random_state,
            n_jobs=-1  # Usa tutti i core disponibili per velocizzare.
        )
        self.best_model = None  # Verrà popolato dopo il tuning o il training.

    def evaluate_baseline(self, X, y, cv_splits=5):

        """Valuta le performance del modello base (senza tuning) usando la Cross-Validation."""

        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(self.base_model, X, y, cv=cv, scoring='accuracy')

        print(f"--- Baseline LightGBM ---")
        print(f"Accuratezza media CV: {scores.mean():.4f} (+/- {scores.std():.4f})")
        return scores.mean()

    def tune_hyperparameters(self, X, y, param_grid=None, cv_splits=5):

        """Esegue una Grid Search per trovare i migliori iperparametri.
        Se non viene passata una griglia, ne usa una di default ottimizzata."""

        if param_grid is None:
            # Griglia di default pensata per i dati tabulari dello Spaceship Titanic.
            param_grid = {
                'num_leaves': [20, 31, 50],
                'max_depth': [5, 7, -1],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 300, 500]
            }

        print("Inizio ottimizzazione degli iperparametri (potrebbe volerci un po')...")
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

        grid_search = GridSearchCV(
            estimator=self.base_model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X, y)
        self.best_model = grid_search.best_estimator_

        print(f"Migliori parametri trovati: {grid_search.best_params_}")
        print(f"Miglior accuratezza in CV con tuning: {grid_search.best_score_:.4f}")

        return self.best_model

    def train(self, X_train, y_train):

        """Addestra il modello finale sull'intero dataset di training.
        Usa il modello ottimizzato se è stato fatto il tuning, altrimenti usa la baseline."""

        if self.best_model is None:
            print("Nessun tuning effettuato. Addestramento con i parametri di default...")
            self.best_model = self.base_model

        self.best_model.fit(X_train, y_train)
        print("Addestramento completato!")

    def predict(self, X_test):

        """Effettua le predizioni sui nuovi dati."""

        if self.best_model is None:
            raise NotFittedError("Devi prima addestrare il modello chiamando train() o tune_hyperparameters()!")

        return self.best_model.predict(X_test)

    def predict_proba(self, X):
        if self.best_model is None:
            raise ValueError("Il modello deve essere addestrato prima.")
        # Prende solo la colonna delle probabilità della classe "True" (Trasportato)
        return self.best_model.predict_proba(X)[:, 1]

# Eccezione per rendere il codice più robusto.
class NotFittedError(Exception):
    pass