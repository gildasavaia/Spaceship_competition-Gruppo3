import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

class LightGBMTrainer:
    """ Classe per gestire l'addestramento, l'ottimizzazione e le predizioni
    del modello LightGBM per lo Spaceship Titanic."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_model = None

    def tune_hyperparameters(self, X, y, param_grid=None, cv_splits=5, fast_mode=True):
        """
        Se fast_mode=True: Addestra istantaneamente con i parametri ottimali scolpiti nel codice (Stile XGBoost).
        Se fast_mode=False: Esegue una ricerca veloce con RandomizedSearchCV per trovare nuovi parametri.
        """
        if fast_mode:
            print("🚀 [Fast Mode] Utilizzo parametri ottimali pre-impostati (Nessuna ricerca!)...")
            # Questi sono parametri eccellenti di default per LightGBM.
            # Se in futuro ne trovi di migliori, modificali qui!
            self.best_model = lgb.LGBMClassifier(
                num_leaves=31,
                max_depth=7,
                learning_rate=0.05,
                n_estimators=300,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.best_model.fit(X, y)
            print("✅ Addestramento completato in un lampo!")
            return self.best_model

        else:
            print("🔍 Inizio ottimizzazione intelligente (RandomizedSearchCV)...")
            base_model = lgb.LGBMClassifier(random_state=self.random_state, n_jobs=-1)

            if param_grid is None:
                param_grid = {
                    'num_leaves': [20, 31, 50, 70],
                    'max_depth': [5, 7, -1],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [100, 300, 500]
                }

            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

            # Usiamo RandomizedSearchCV al posto di GridSearchCV: proverà solo 10 combinazioni casuali
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
            print(f"Miglior accuratezza in CV con tuning: {random_search.best_score_:.4f}")
            return self.best_model

    def predict(self, X_test):
        """Effettua le predizioni sui nuovi dati."""
        if self.best_model is None:
            raise Exception("Devi prima addestrare il modello!")
        return self.best_model.predict(X_test)

    def predict_proba(self, X):
        """Restituisce le probabilità."""
        if self.best_model is None:
            raise Exception("Devi prima addestrare il modello!")
        return self.best_model.predict_proba(X)[:, 1]