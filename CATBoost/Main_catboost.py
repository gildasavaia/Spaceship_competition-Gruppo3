
from sklearn.model_selection import train_test_split
from Model_CatBoost import *
from CATBoost.Model_CatBoost import prepare_test


train_path = "../data/preprocessed_folds/holdout_tree_train.csv"
test_path = "../data/preprocessed_folds/holdout_tree_test.csv"



train_df, test_df = load_data(train_path, test_path)


#
X, y = prepare_data(train_df)
X_test = prepare_test(test_df)


#
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)



model = create_catboost_model()



train_model(model, X_train, y_train, X_val, y_val)



evaluate_model(model, X_val, y_val)



predictions = predict(model, X_test)


show_predictions(test_df, predictions, n=10)




