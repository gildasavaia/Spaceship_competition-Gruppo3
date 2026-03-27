from Rete_neurale_model import *

# ---------------------------
# 1. Caricamento dati
# ---------------------------
train_df, test_df = load_data(
    "train_processed_example.csv",
    "test_processed_example.csv"
)

# ---------------------------
#  Preparazione + scaling
# ---------------------------
X, y, scaler = prepare_data(train_df)
X_test = prepare_test(test_df, scaler)

# ---------------------------
#  Creazione modello NN
# ---------------------------
model = create_nn_model()

# ---------------------------
#  Training
# ---------------------------
train_model(model, X, y)

# ---------------------------
#  Valutazione
# ---------------------------
accuracy = evaluate_model(model, X, y)
print(f"\n Accuracy media (NN): {accuracy:.4f}")

# ---------------------------
#  Predizione
# ---------------------------
predictions = predict(model, X_test)

# ---------------------------
#  Mostra predizioni
# ---------------------------
show_predictions(test_df, predictions)

# ---------------------------
#  Submission
# ---------------------------
create_submission(test_df, predictions)
print("\n Submission NN creata")