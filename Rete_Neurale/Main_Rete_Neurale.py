from Rete_neurale_model import *

# ---------------------------
# 1. Caricamento dati
# ---------------------------
train_path = "../Dataset_Giocattolo/train_processed_example.csv"
test_path = "../Dataset_Giocattolo/test_processed_example.csv"

train_df, test_df = load_data(train_path, test_path)

# ---------------------------
# 2. Preparazione dati
# ---------------------------
X, y = prepare_data(train_df)
X_test = prepare_test(test_df)

# ⚠️ Se NON hai già fatto encoding
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# ---------------------------
# 3. Creazione modello (Pipeline)
# ---------------------------
model = create_pipeline_model()

# ---------------------------
# 4. Valutazione
# ---------------------------
accuracy = evaluate_model(model, X, y)
print(f"\n📊 Accuracy media (NN): {accuracy:.4f}")

# ---------------------------
# 5. Training finale su tutti i dati
# ---------------------------
model = train_model(model, X, y)

# ---------------------------
# 6. Predizione
# ---------------------------
predictions = predict(model, X_test)

# ---------------------------
# 7. Mostra predizioni
# ---------------------------
show_predictions(test_df, predictions)

# ---------------------------
# 8. Submission
# ---------------------------
create_submission(test_df, predictions)

print("\n✅ Submission NN creata")