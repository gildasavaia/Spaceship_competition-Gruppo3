from Development import *

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
# 3. Creazione modello
# ---------------------------
model = create_model() # oppure create_model() per versione base

# ---------------------------
# 4. Addestramento modello
# ---------------------------
train_model(model, X, y)

# ---------------------------
# 5. Valutazione modello
# ---------------------------
accuracy = evaluate_model(model, X, y)
print(f"\n Accuracy media (cross-validation): {accuracy:.4f}")

# ---------------------------
# 6. Predizione sul test set
# ---------------------------
predictions = predict(model, X_test)

# ---------------------------
# 7. Mostra prime predizioni
# ---------------------------
show_predictions(test_df, predictions, n=10)




# ---------------------------
# 9. Creazione submission
# ---------------------------
create_submission(test_df, predictions)
print("\n Submission creata: submission.csv")