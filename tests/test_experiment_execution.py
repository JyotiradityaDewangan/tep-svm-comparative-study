import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from openpyxl import load_workbook

import config
from src.experiments import run_svm_experiments
from src.preprocessing import fit_preprocessor_on_training as real_fit_preprocessor
from src.run_reporting import write_run_records
import json


class ExperimentExecutionTests(unittest.TestCase):
    def test_individual_experiment_does_not_run_others_or_replace_combined_workbook(self):
        rng = np.random.default_rng(12)
        y = np.tile([0, 1], 50)
        X = rng.normal(size=(100, 3))
        X[:, 0] += y
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "results"
            combined = root / "svm_comparison" / "svm_experiments.xlsx"
            combined.parent.mkdir(parents=True)
            combined.write_bytes(b"historical combined workbook sentinel")
            with patch.object(config, "RESULTS_DIR", root), \
                 patch.object(config, "CSV_DIR", root / "csv_reports"), \
                 patch.object(config, "EXCEL_DIR", root / "excel_reports"), \
                 patch.object(config, "SAVE_MODEL", False), \
                 patch.object(config, "EXPERIMENT_MAX_ROWS", 40):
                result = run_svm_experiments(X, y, selected="c")
            self.assertEqual(set(result), {"c_parameter", "_run_metadata"})
            self.assertTrue((root / "c_parameter_experiment" / "linear_svc_c_parameter.csv").is_file())
            self.assertFalse((root / "kernel_experiment").exists())
            self.assertFalse((root / "convergence").exists())
            self.assertFalse((root / "svm_from_scratch").exists())
            self.assertEqual(combined.read_bytes(), b"historical combined workbook sentinel")

    def test_all_svm_experiments_write_independent_outputs_and_complete_workbook(self):
        rng = np.random.default_rng(71)
        group_count = 40
        rows_per_group = 8
        y = np.tile(np.array([0, 1] * 4), group_count)
        groups = np.repeat(np.arange(group_count), rows_per_group)
        X = rng.normal(size=(len(y), 5))
        X[:, 0] += 1.2 * y
        X[::17, 2] = np.nan

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "results"
            patchers = [
                patch.object(config, "RESULTS_DIR", root),
                patch.object(config, "CSV_DIR", root / "csv_reports"),
                patch.object(config, "EXCEL_DIR", root / "excel_reports"),
                patch.object(config, "PLOTS_DIR", root / "plots"),
                patch.object(config, "MODEL_DIR", root / "models"),
                patch.object(config, "SAVE_MODEL", False),
                patch.object(config, "EXPERIMENT_MAX_ROWS", 80),
                patch.object(config, "EXPERIMENT_SAMPLE_SIZES", [20, 40]),
                patch.object(config, "SCRATCH_COMPARE_ROWS", 80),
                patch.object(config, "QP_MAX_ROWS", 80),
                patch.object(config, "SCRATCH_ITERATIONS", 15),
            ]
            preprocessing_calls = []

            def track_preprocessing(fit_raw, validation_raw):
                result = real_fit_preprocessor(fit_raw, validation_raw)
                X_fit, _, imputer, scaler = result
                expected_medians = np.nanmedian(np.asarray(fit_raw, dtype=float), axis=0)
                np.testing.assert_allclose(imputer.statistics_, expected_medians)
                if scaler is not None:
                    expected_scaled_mean = imputer.transform(np.asarray(fit_raw, dtype=float)).mean(axis=0)
                    np.testing.assert_allclose(scaler.mean_, expected_scaled_mean, atol=1e-12)
                preprocessing_calls.append((len(fit_raw), None if validation_raw is None else len(validation_raw)))
                return result
            with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4], \
                 patchers[5], patchers[6], patchers[7], patchers[8], patchers[9], patchers[10], \
                 patch("src.experiments.fit_preprocessor_on_training", side_effect=track_preprocessing):
                results = run_svm_experiments(X, y, groups, selected="all")

            expected = {"implementations", "convergence", "c_parameter", "kernel", "sample_size", "_run_metadata"}
            self.assertEqual(set(results), expected)
            self.assertEqual(results["c_parameter"]["C"].tolist(), [0.1, 1.0, 10.0, 100.0])
            self.assertEqual(len(results["convergence"]), 15)
            self.assertEqual(set(results["kernel"]["kernel"]), {"linear", "rbf"})
            self.assertEqual(results["_run_metadata"]["split_strategy"],
                             "group-aware split by source file and simulationRun")
            self.assertEqual(results["_run_metadata"]["validation_rows"], 64)
            self.assertEqual(len(results["implementations"]), 3)
            self.assertGreaterEqual(len(preprocessing_calls), 5)

            required = [
                "svm_comparison/linear_implementation_comparison.csv",
                "convergence/scratch_svm_loss.csv",
                "c_parameter_experiment/linear_svc_c_parameter.csv",
                "kernel_experiment/linear_vs_rbf.csv",
                "svm_comparison/sample_size_fit_time.csv",
                "svm_comparison/svm_experiments.xlsx",
            ]
            for rel in required:
                self.assertTrue((root / rel).is_file(), rel)
            c_csv = pd.read_csv(root / "c_parameter_experiment" / "linear_svc_c_parameter.csv")
            np.testing.assert_allclose(c_csv["accuracy"], results["c_parameter"]["accuracy"])
            c_workbook = load_workbook(root / "c_parameter_experiment" / "linear_svc_c_parameter.xlsx",
                                       read_only=True, data_only=True)
            try:
                c_rows = list(c_workbook["C Experiment"].values)
                np.testing.assert_allclose([row[1] for row in c_rows[1:]], c_csv["accuracy"])
            finally:
                c_workbook.close()
            workbook = load_workbook(root / "svm_comparison" / "svm_experiments.xlsx", read_only=True)
            try:
                self.assertEqual(set(workbook.sheetnames),
                                 {"Implementations", "Convergence", "C Parameter", "Kernel", "Sample Size"})
            finally:
                workbook.close()

            prepared = {"y_train": y[:200], "y_test": y[200:],
                        "split_strategy": "synthetic group-aware test"}
            dataset_summary = {"dataset_name": "Synthetic test data (not TEP)",
                "source_files": ["synthetic_normal.csv", "synthetic_faulty.csv"],
                "sampled_observations": len(y), "rows_scanned": len(y),
                "rows_by_source_file": {"synthetic_normal.csv": 160, "synthetic_faulty.csv": 160},
                "sampled_class_distribution_by_source": {}, "input_features": ["a", "b", "c", "d", "e"],
                "metadata_columns": [], "internal_columns_excluded": []}
            with patch.object(config, "RESULTS_DIR", root):
                manifest_path, report_path = write_run_records(
                    "unit_test_run", dataset_summary, prepared,
                    main_metrics={"accuracy": 0.75}, experiments=results,
                    run_kind="synthetic-test", confusion_matrix=[[2, 0], [1, 1]],
                    input_hashes={"training_csvs": {"synthetic.csv": {"sha256": "abc"}}})
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["run_id"], "unit_test_run")
            self.assertIn("implementations", manifest["metrics"]["experiments"])
            self.assertIsInstance(manifest["metrics"]["main_model"], dict)
            self.assertEqual(manifest["metrics"]["main_model"]["accuracy"], 0.75)
            self.assertEqual(manifest["metrics"]["main_model"]["confusion_matrix"], [[2, 0], [1, 1]])
            self.assertEqual(manifest["input_file_hashes"]["training_csvs"]["synthetic.csv"]["sha256"], "abc")
            self.assertTrue(report_path.is_file())
            self.assertIn("fitting only", report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
