import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import config
from src.svm_model import make_model, save_artifacts, load_artifacts
from src.prediction import predict_file


class ModelPredictionTests(unittest.TestCase):
    def test_main_svc_save_load_and_new_csv_prediction(self):
        X, y = make_classification(n_samples=160, n_features=6, n_informative=4,
                                   n_redundant=1, random_state=9)
        names = [f"xmeas_{index}" for index in range(1, 7)]
        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.25, random_state=12, stratify=y)
        X_train_frame = pd.DataFrame(X_train, columns=names)
        X_test_frame = pd.DataFrame(X_test, columns=names)
        imputer = SimpleImputer(strategy="median").fit(X_train_frame)
        scaler = StandardScaler().fit(imputer.transform(X_train_frame))
        X_train_scaled = scaler.transform(imputer.transform(X_train_frame))
        X_test_scaled = scaler.transform(imputer.transform(X_test_frame))
        model = make_model().fit(X_train_scaled, y_train)
        expected = model.predict(X_test_scaled)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_dir = root / "models"
            predictions_dir = root / "predictions"
            with patch.object(config, "MODEL_DIR", model_dir), \
                 patch.object(config, "PREDICTIONS_DIR", predictions_dir):
                save_artifacts(model, imputer, scaler, names)
                loaded, loaded_imputer, loaded_scaler, loaded_names = load_artifacts()
                actual = loaded.predict(loaded_scaler.transform(loaded_imputer.transform(X_test_frame)))
                np.testing.assert_array_equal(actual, expected)

                sensor_csv = root / "new_readings.csv"
                output_csv = predictions_dir / "predictions.csv"
                pd.DataFrame(X_test[:7], columns=names).to_csv(sensor_csv, index=False)
                output = predict_file(sensor_csv, output_csv, model_dir=model_dir)
                self.assertEqual(len(output), 7)
                self.assertTrue(set(output["predicted_class"]).issubset({"NORMAL", "FAULTY"}))
                self.assertTrue(output_csv.is_file())
                self.assertEqual(loaded_names, names)


if __name__ == "__main__":
    unittest.main()
