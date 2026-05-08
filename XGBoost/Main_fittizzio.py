from Development import *

# ---------------------------
# 1. Caricamento dati
# ---------------------------
train_path = "../data/preprocessed_folds/processed_full_tree.csv"
test_path = "../data/preprocessed_folds/processed_full_tree_test.csv"
yt_2 = "../data/sample_submission.csv"
train_df, test_df = load_data(train_path, test_path)
# ---------------------------
# 2. Caricamento label reali per test (yt_2)
# ---------------------------
yt_2_df = pd.read_csv(yt_2)



# aggiunta colonna al test set
test_df = test_df.reset_index(drop=True)
yt_2_df = yt_2_df.reset_index(drop=True)

test_df["Transported"] = yt_2_df["Transported"]

# y_test (label reali)
y_test = test_df["Transported"]

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
