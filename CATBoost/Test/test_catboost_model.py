import unittest
import pandas as pd
import numpy as np
import os

from CATBoost.Model_CatBoost import (
    prepare_test,
    create_catboost_model,
    train_model,
    predict,
    evaluate_model,
    create_submission
)


class TestCatBoostModel(unittest.TestCase):

    def setUp(self):
        """Crea dati finti ma più robusti"""
        np.random.seed(42)

        n_samples = 20  # 👈 più dati → CV stabile

        self.train_df = pd.DataFrame({
            "PassengerId": [f"{i:04d}" for i in range(n_samples)],
            "Feature1": np.random.randint(0, 10, n_samples),
            "Feature2": np.random.randint(0, 2, n_samples),
            "Transported": np.random.choice([True, False], n_samples)
        })

        self.test_df = pd.DataFrame({
            "PassengerId": ["1001", "1002", "1003"],
            "Feature1": [5, 6, 7],
            "Feature2": [1, 0, 1]
        })

        self.X = self.train_df.drop(["Transported", "PassengerId"], axis=1)
        self.y = self.train_df["Transported"]
        self.X_test = self.test_df.drop("PassengerId", axis=1)



    def test_prepare_test(self):
        X_test = prepare_test(self.test_df)
        self.assertNotIn("PassengerId", X_test.columns)



    def test_create_model(self):
        model = create_catboost_model()
        self.assertIsNotNone(model)



    def test_train_and_predict(self):
        model = create_catboost_model()
        model = train_model(model, self.X, self.y)

        preds = predict(model, self.X_test)

        self.assertEqual(len(preds), len(self.X_test))



    def test_evaluate_model(self):
        model = create_catboost_model()
        score = evaluate_model(model, self.X, self.y, cv=3)

        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)



    def test_create_submission(self):
        predictions = np.array([True, False, True])
        output_path = "test_submission.csv"

        create_submission(self.test_df, predictions, output_path)

        self.assertTrue(os.path.exists(output_path))

        df = pd.read_csv(output_path)

        self.assertIn("PassengerId", df.columns)
        self.assertIn("Transported", df.columns)
        self.assertEqual(len(df), len(self.test_df))

        # 🧹 cleanup (importantissimo)
        os.remove(output_path)


if __name__ == '__main__':
    unittest.main()