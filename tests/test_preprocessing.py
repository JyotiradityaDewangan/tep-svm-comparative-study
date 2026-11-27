import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import config
from src.data_loader import load_dataset
from src.preprocessing import fit_preprocessor_on_training
from src.experiments import _inner_split


class PreprocessingTests(unittest.TestCase):
    def test_imputer_and_scaler_fit_only_on_fit_rows(self):
        fit_raw = np.array([[0.0, 10.0], [2.0, 20.0], [np.nan, 30.0]])
        validation_raw = np.array([[1000.0, 2000.0], [np.nan, 3000.0]])
        X_fit, X_validation, imputer, scaler = fit_preprocessor_on_training(
            fit_raw, validation_raw)

        np.testing.assert_allclose(imputer.statistics_, [1.0, 20.0])
        np.testing.assert_allclose(scaler.mean_, [1.0, 20.0])
        expected_validation = scaler.transform(imputer.transform(validation_raw))
        np.testing.assert_allclose(X_validation, expected_validation)
        self.assertEqual(X_fit.shape, (3, 2))
        self.assertGreater(abs(X_validation[0, 0]), 100.0)

    def test_group_split_separates_runs(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(240, 4))
        y = np.tile(np.array([0, 1, 0, 1, 0, 1, 0, 1]), 30)
        groups = np.repeat(np.arange(30), 8)
        *_, dev_idx, val_idx, method = _inner_split(X, y, groups)
        self.assertIn("group-aware", method)
        self.assertTrue(set(groups[dev_idx]).isdisjoint(set(groups[val_idx])))

    def test_loader_excludes_metadata_and_records_source_class_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_dir = root / "raw"
            data_dir.mkdir()
            normal = pd.DataFrame({"faultNumber": [0, 0], "simulationRun": [1, 1],
                                   "sample": [1, 2], "xmeas_1": [1.0, 2.0]})
            faulty = pd.DataFrame({"faultNumber": [1, 2], "simulationRun": [2, 2],
                                   "sample": [1, 2], "xmeas_1": [3.0, 4.0]})
            normal.to_csv(data_dir / "TEP_FaultFree_Training.csv", index=False)
            faulty.to_csv(data_dir / "TEP_Faulty_Training.csv", index=False)
            with patch.object(config, "DATASET_PATH", data_dir), \
                 patch.object(config, "PROCESSED_DIR", root / "processed"):
                data, features, summary = load_dataset()
            self.assertEqual(features, ["xmeas_1"])
            self.assertEqual(set(data["_binary_target"]), {0, 1})
            self.assertEqual(summary["sampled_class_distribution_by_source"]["TEP_Faulty_Training.csv"]["FAULTY"], 2)


if __name__ == "__main__":
    unittest.main()
