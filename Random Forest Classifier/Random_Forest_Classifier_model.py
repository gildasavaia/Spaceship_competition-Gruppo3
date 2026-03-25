from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

class RandomForestTrainer:
    """
    Classe per gestire l'addestramento, l'ottimizzazione e le predizioni
    del modello Random Forest per lo Spaceship Titanic.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        # Inizializziamo il modello base
        self.base_model = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1  # Usa tutti i core disponibili
        )
        self.best_model = None

    def evaluate_baseline(self, X, y, cv_splits=5):
        # NOTA: Se usi il mini-dataset da 80 righe, metti cv_splits=2 per evitare errori
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(self.base_model, X, y, cv=cv, scoring='accuracy')

        print(f"--- Baseline Random Forest ---")
        print(f"Accuratezza media CV: {scores.mean():.4f} (+/- {scores.std():.4f})")
        return scores.mean()

    def tune_hyperparameters(self, X, y, param_grid=None, cv_splits=5):
        """Esegue la Grid Search specifica per Random Forest."""
        if param_grid is None:
            # Griglia di parametri tipica per la Random Forest
            param_grid = {
                'n_estimators': [100, 200, 300],  # Numero di alberi
                'max_depth': [None, 10, 20],  # Profondità massima (None = infinita)
                'min_samples_split': [2, 5, 10],  # Minimo di campioni per dividere un nodo
                'min_samples_leaf': [1, 2, 4]  # Minimo di campioni in una foglia finale
            }

        print("Inizio ottimizzazione dei parametri Random Forest...")
        # NOTA: Usa cv=2 se stai testando il mini-dataset, altrimenti lascia cv_splits
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
        print(f"Miglior accuratezza CV con tuning: {grid_search.best_score_:.4f}")

        return self.best_model

    def train(self, X_train, y_train):
        if self.best_model is None:
            print("Nessun tuning effettuato. Uso i parametri di default...")
            self.best_model = self.base_model

        self.best_model.fit(X_train, y_train)
        print("Addestramento completato!")

    def predict(self, X_test):
        if self.best_model is None:
            raise Exception("Devi prima addestrare il modello!")
        return self.best_model.predict(X_test)