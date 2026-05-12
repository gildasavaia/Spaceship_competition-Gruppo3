from Model_CatBoost import (
    load_data,
    prepare_data,
    prepare_test,
    create_catboost_model,
    train_model,
    evaluate_model,
    predict
)

# =========================
# PATHS
# =========================

train_path = "../data/preprocessed_folds/processed_full_tree.csv"
test_path = "../data/preprocessed_folds/processed_full_tree_test.csv"


# =========================
# LOAD DATA
# =========================

train_df, test_df = load_data(train_path, test_path)

X, y = prepare_data(train_df)
X_test = prepare_test(test_df)


# =========================
# TEST LABEL (se presente)
# =========================

y_test = test_df["Transported"] if "Transported" in test_df.columns else None


# =========================
# MODEL
# =========================

model = create_catboost_model()


# =========================
# TRAIN (NO SPLIT - HOLDOUT READY)
# =========================

train_model(model, X, y)


# =========================
# EVALUATION SU TEST SET
# =========================

if y_test is not None:
    evaluate_model(model, X_test, y_test)


# =========================
# PREDICTIONS
# =========================

predictions = predict(model, X_test)