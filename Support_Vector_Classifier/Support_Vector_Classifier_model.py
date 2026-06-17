from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class SupportVectorTrainer:
    def __init__(self, random_state=42):
        # IL TRUCCO È QUI: Creiamo una pipeline automatica.
        # Ogni volta che passiamo dei dati, Python prima li "schiaccerà" (StandardScaler)
        # mettendo tutte le variabili sulla stessa scala geometrica, e poi li passerà al SVC.
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(random_state=random_state, probability=True))
        ])
        self.best_model = None

    def tune_hyperparameters(self, X, y):
        # Noterai il prefisso 'svc__' prima dei parametri.
        # Serve per dire a Python che questi parametri sono per l'SVC e non per lo Scaler.
        param_grid = {
            'svc__C': [0.1, 1, 10],
            'svc__kernel': ['rbf', 'poly'],
            'svc__gamma': ['scale', 'auto']
        }

        print("Inizio ricerca iperparametri per SVC (può richiedere tempo)...")
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X, y)
        self.best_model = grid_search.best_estimator_
        print(f"Migliori parametri trovati: {grid_search.best_params_}")
        print(f"Miglior punteggio Cross-Validation: {grid_search.best_score_:.4f}")

    def predict(self, X):
        if self.best_model is None:
            raise ValueError("Il modello deve essere addestrato prima di fare predizioni.")
        # Quando chiami predict, lo StandardScaler interno scalerà in automatico
        # anche i dati di test usando le regole imparate sui dati di train!
        return self.best_model.predict(X)

    def predict_proba(self, X):
        if self.best_model is None:
            raise ValueError("Il modello deve essere addestrato prima.")
        # Prende solo la colonna delle probabilità della classe "True" (Trasportato)
        return self.best_model.predict_proba(X)[:, 1]