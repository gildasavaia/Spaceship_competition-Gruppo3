import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

class RandomForestTrainer:
    """ Classe per gestire l'addestramento, l'ottimizzazione e le predizioni
    del modello Random Forest per lo Spaceship Titanic."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_model = None

    def tune_hyperparameters(self, X, y, param_grid=None, cv_splits=5, fast_mode=True):
        """
        Ottimizza gli iperparametri o usa parametri ottimali predefiniti.
        """
        if fast_mode:
            print("🚀 [Fast Mode] Utilizzo parametri Random Forest ottimali pre-impostati...")
            # Parametri bilanciati per evitare overfitting ma catturare la complessità
            self.best_model = RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.best_model.fit(X, y)
            print("✅ Addestramento Random Forest completato!")
            return self.best_model

        else:
            print("🔍 Inizio ottimizzazione intelligente (RandomizedSearchCV) per RF...")
            base_model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)

            if param_grid is None:
                param_grid = {
                    'n_estimators': [100, 300, 500],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }

            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=10,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state
            )

            random_search.fit(X, y)
            self.best_model = random_search.best_estimator_

            print(f"✅ Migliori parametri trovati: {random_search.best_params_}")
            return self.best_model

    def predict(self, X_test):
        return self.best_model.predict(X_test)

    def predict_proba(self, X):
        return self.best_model.predict_proba(X)[:, 1]