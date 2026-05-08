from Rete_neurale_model import *


#  Caricamento dati

train_path = "../data/preprocessed_folds/holdout_nn_train.csv"
test_path = "../data/preprocessed_folds/holdout_nn_test.csv"

# 1. Caricamento dati
train_df, test_df = load_data(train_path, test_path)

# 2. Preparazione
X, y = prepare_data(train_df)
X_test = prepare_test(test_df)

# 3. Encoding
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# 4. Creazione modello
model = create_pipeline_model()

# 5. FIT PRIMA DI TUTTO
model = train_model(model, X, y)

# 6. VALUTAZIONE
accuracy = evaluate_on_test(model, X, y)
print(f"\nAccuracy media (NN): {accuracy:.4f}")

# 7. PREDIZIONE
predictions = predict(model, X_test)

# 8. OUTPUT
show_predictions(test_df, predictions)