import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import config
from src.pca_experiment import fit_pca_training_split, run_pca_experiment
from src.svm_model import make_model


class PCAExperimentTests(unittest.TestCase):
    def test_pca_fit_scope_variance_and_transform_alignment(self):
        rng = np.random.default_rng(24)
        X_train = rng.normal(size=(180, 8))
        X_train[:, 7] = X_train[:, 0] * 0.2 + X_train[:, 1] * 0.1
        X_test = rng.normal(loc=50.0, size=(30, 8))
        pca, train_pca, test_pca = fit_pca_training_split(X_train, X_test, 0.95)

        np.testing.assert_allclose(pca.mean_, X_train.mean(axis=0))
        self.assertGreaterEqual(pca.explained_variance_ratio_.sum(), 0.95)
        expected_count = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.95) + 1)
        self.assertEqual(train_pca.shape[1], expected_count)
        np.testing.assert_allclose(test_pca, pca.transform(X_test))
        self.assertEqual(train_pca.shape[0], len(X_train))
        self.assertEqual(test_pca.shape[0], len(X_test))

    def test_run_uses_rbf_saves_four_plots_and_comparison_table(self):
        rng = np.random.default_rng(18)
        X = rng.normal(size=(120, 5))
        y = np.tile([0, 1], 60)
        X[:, 0] += y * 0.7
        with tempfile.TemporaryDirectory() as temp:
            result_root = Path(temp) / "results"
            with patch.object(config, "RESULTS_DIR", result_root), \
                 patch.object(config, "SAVE_EXCEL_REPORTS", True), \
                 patch.object(config, "SVM_KERNEL", "linear"), \
                 patch("src.pca_experiment.make_model", wraps=make_model) as model_factory:
                table = run_pca_experiment(X[:90], y[:90], X[90:], y[90:])
            self.assertEqual(len(table), 2)
            self.assertEqual(model_factory.call_count, 2)
            self.assertTrue(all(call.kwargs.get("kernel") == "rbf"
                                for call in model_factory.call_args_list))
            folder = result_root / "pca"
            plots = ["explained_variance.png", "cumulative_explained_variance.png",
                     "dimensionality_reduction.png", "pca_svm_performance_comparison.png"]
            for name in plots:
                self.assertTrue((folder / name).is_file(), name)
            saved = pd.read_csv(folder / "pca_svm_comparison.csv")
            self.assertEqual(saved["features"].tolist(), table["features"].tolist())
            self.assertTrue((folder / "pca_svm_comparison.xlsx").is_file())


if __name__ == "__main__":
    unittest.main()
