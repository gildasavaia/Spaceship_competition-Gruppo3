from Development import *

# ---------------------------
# 1. Caricamento dati
# ---------------------------
train_path = "processed_full_tree.csv"
test_path = "holdout_tree_test.csv"  # test separato

train_df, test_df = load_data(train_path, test_path)

# ---------------------------
# 2. Preparazione dati
# ---------------------------
X, y = prepare_data(train_df)
X_test = prepare_test(test_df)

# Se il test ha la label (nel tuo caso sì)
y_test = test_df["Transported"] if "Transported" in test_df.columns else None

# ---------------------------
# 3. Creazione modello
# ---------------------------
model = create_model()

# ---------------------------
# 4. Addestramento modello
# ---------------------------
train_model(model, X, y)

# ---------------------------
# 5. Valutazione REALE (NO cross-validation)
# ---------------------------
if y_test is not None:
    evaluate_on_test(model, X_test, y_test)

# ---------------------------
# 6. Predizione sul test set
# ---------------------------
predictions = predict(model, X_test)

# ---------------------------
# 7. Mostra prime predizioni
# ---------------------------
show_predictions(predictions, n=10)

# ---------------------------
# 8. Feature importance
# ---------------------------


# ---------------------------
# 9. Creazione submission
# ---------------------------
