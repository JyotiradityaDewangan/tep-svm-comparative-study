import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.official_testing import evaluate_official_files, _score_saved_rbf


class OfficialTestingTests(unittest.TestCase):
    def test_saved_artifacts_score_every_official_row_without_refit(self):
        rng = np.random.default_rng(101)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "saved_model"
            test_dir = root / "testing"
            output_dir = root / "output"
            training_dir = root / "training"
            for folder in (model_dir, test_dir, training_dir):
                folder.mkdir()

            feature_names = [f"xmeas_{index}" for index in range(1, 42)] + [
                f"xmv_{index}" for index in range(1, 12)]
            X_train = rng.normal(size=(80, 52))
            y_train = np.array([0] * 40 + [1] * 40)
            X_train[40:, 0] += 2.0
            training_frame = pd.DataFrame(X_train, columns=feature_names)
            imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(training_frame)
            scaled_train = imputer.transform(training_frame)
            scaler = StandardScaler().fit(scaled_train)
            model = SVC(kernel="rbf", C=1, gamma="scale").fit(
                scaler.transform(scaled_train), y_train)
            probe = scaler.transform(imputer.transform(training_frame.iloc[:13]))
            np.testing.assert_allclose(_score_saved_rbf(model, probe),
                                       model.decision_function(probe), rtol=1e-10, atol=1e-10)
            original_support = model.support_vectors_.copy()
            for name, artifact in (("imputer.pkl", imputer), ("scaler.pkl", scaler),
                                   ("trained_svm_model.pkl", model),
                                   ("feature_names.pkl", feature_names)):
                joblib.dump(artifact, model_dir / name)

            free = pd.DataFrame(rng.normal(size=(3, 52)), columns=feature_names)
            free.insert(0, "sample", [1, 2, 3])
            free.insert(0, "simulationRun", [1, 1, 1])
            free.insert(0, "faultNumber", [0, 0, 0])
            faulty = pd.DataFrame(rng.normal(size=(4, 52)), columns=feature_names)
            faulty.insert(0, "sample", [1, 2, 3, 4])
            faulty.insert(0, "simulationRun", [1, 1, 1, 1])
            faulty.insert(0, "faultNumber", [1, 1, 1, 1])
            free.to_csv(test_dir / "TEP_FaultFree_Testing.csv", index=False)
            faulty.to_csv(test_dir / "TEP_Faulty_Testing.csv", index=False)

            metrics, saved_to = evaluate_official_files(
                model_dir, testing_dir=test_dir, output_dir=output_dir,
                chunk_rows=2, training_dir=training_dir, run_id="synthetic")
            self.assertEqual(saved_to, output_dir)
            self.assertEqual(metrics["observations"], 7)
            self.assertEqual(metrics["feature_count"], 52)
            self.assertEqual(metrics["normal_observations"], 3)
            self.assertEqual(metrics["faulty_observations"], 4)
            np.testing.assert_array_equal(joblib.load(model_dir / "trained_svm_model.pkl").support_vectors_,
                                          original_support)
            self.assertTrue((output_dir / "official_test_metrics.csv").is_file())
            self.assertTrue((output_dir / "confusion_matrix.png").is_file())
            manifest = json.loads((output_dir / "official_test_manifest.json").read_text())
            self.assertEqual(manifest["run_id"], "synthetic")
            self.assertEqual(len(manifest["files"]), 2)
            self.assertEqual(len(manifest["files"]["TEP_Faulty_Testing.csv"]["sha256"]), 64)

    def test_missing_required_feature_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir, test_dir = root / "models", root / "testing"
            model_dir.mkdir(); test_dir.mkdir()
            features = [f"xmeas_{index}" for index in range(1, 42)] + [
                f"xmv_{index}" for index in range(1, 12)]
            X = np.vstack([np.zeros(52), np.ones(52)])
            model = SVC(kernel="rbf").fit(X, [0, 1])
            joblib.dump(model, model_dir / "trained_svm_model.pkl")
            joblib.dump(SimpleImputer().fit(X), model_dir / "imputer.pkl")
            joblib.dump(None, model_dir / "scaler.pkl")
            joblib.dump(features, model_dir / "feature_names.pkl")
            short = pd.DataFrame({"faultNumber": [0], **{name: [0.0] for name in features[:-1]}})
            for filename in ("TEP_FaultFree_Testing.csv", "TEP_Faulty_Testing.csv"):
                short.to_csv(test_dir / filename, index=False)
            with self.assertRaisesRegex(ValueError, "missing saved model features"):
                evaluate_official_files(model_dir, testing_dir=test_dir,
                                        output_dir=root / "output", training_dir=root / "empty")


if __name__ == "__main__":
    unittest.main()
