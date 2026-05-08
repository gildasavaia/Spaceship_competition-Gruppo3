from sklearn.model_selection import train_test_split

from Model_CatBoost import (
    load_data,
    prepare_data,
    prepare_test,
    create_catboost_model,
    train_model,
    evaluate_model,
    predict,


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
# SPLIT
# =========================

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# =========================
# MODEL
# =========================

model = create_catboost_model()


# =========================
# TRAIN
# =========================

train_model(model, X_train, y_train, X_val, y_val)


# =========================
# EVAL
# =========================

evaluate_model(model, X_val, y_val)





# =========================
# PREDICT TEST
# =========================

predictions = predict(model, X_test)


