from Model_CatBoost import *
from sklearn.model_selection import train_test_split

# ---------------------------
# 1. Caricamento dati
# ---------------------------
train_path = "train_processed_example.csv"
test_path = "test_processed_example.csv"

train_df, test_df = load_data(train_path, test_path)

# ---------------------------
# 2. Preparazione dati
# ---------------------------
X, y = prepare_data(train_df)
X_test = prepare_test(test_df)

# ---------------------------
# 3. Divisione train/validation per early stopping
# ---------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 4. Creazione modello CatBoost
# ---------------------------
model = create_catboost_model()

# ---------------------------
# 5. Training con early stopping
# ---------------------------
train_model(model, X_train, y_train, X_val, y_val)

# ---------------------------
# 6. Valutazione con cross-validation
# ---------------------------
accuracy = evaluate_model(model, X, y)
print(f"\n Accuracy media (cross-validation): {accuracy:.4f}")

# ---------------------------
# 7. Predizione sul test set
# ---------------------------
predictions = predict(model, X_test)

# ---------------------------
# 8. Mostra prime predizioni
# ---------------------------
show_predictions(test_df, predictions, n=10)

# ---------------------------
# 9. Analisi feature importance
# ---------------------------
plot_feature_importance(model, X.columns)

# ---------------------------
# 10. Creazione submission
# ---------------------------
create_submission(test_df, predictions)
print("\n Submission creata: submission_catboost.csv")

# ---------------------------
# 11. Controllo consistenza colonne
# ---------------------------
