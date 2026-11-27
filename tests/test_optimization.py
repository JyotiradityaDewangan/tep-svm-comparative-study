import importlib.util
import unittest

import numpy as np
from sklearn.svm import SVC

from src.optimization import solve_linear_svm_qp, predict_linear_svm_qp


@unittest.skipUnless(importlib.util.find_spec("cvxopt"), "CVXOPT is not installed")
class CVXOPTValidationTests(unittest.TestCase):
    def test_class_weighted_dual_uses_class_specific_upper_bounds(self):
        X = np.array([[-2.0], [-1.0], [-0.3], [0.2], [1.0], [2.0]])
        y = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0])
        result = solve_linear_svm_qp(X, y, C=2.0, class_weight="balanced")
        np.testing.assert_allclose(result["upper_bounds"], [1.5, 1.5, 1.5, 1.5, 3.0, 3.0])
        self.assertLess(result["diagnostics"]["max_alpha_bound_violation"], 1e-7)

    def test_dual_constraints_objectives_kkt_and_library_predictions(self):
        X = np.array([[-2.0, 0.0], [-1.0, 0.2], [-0.8, -0.1],
                      [0.7, 0.0], [1.0, -0.2], [2.0, 0.1]])
        y = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        result = solve_linear_svm_qp(X, y, C=1.0, class_weight="balanced")
        diag = result["diagnostics"]

        self.assertEqual(result["status"], "optimal")
        self.assertTrue(np.all(result["alpha"] >= -1e-7))
        self.assertTrue(np.all(result["alpha"] <= result["upper_bounds"] + 1e-7))
        self.assertAlmostEqual(float(y @ result["alpha"]), 0.0, delta=1e-6)
        self.assertTrue(np.isfinite(result["weights"]).all())
        self.assertTrue(np.isfinite(result["bias"]))
        self.assertGreaterEqual(diag["primal_objective"], diag["dual_objective"] - 1e-6)
        self.assertLess(abs(diag["duality_gap"]), 1e-4)
        self.assertLess(diag["max_alpha_bound_violation"], 1e-7)
        self.assertLess(diag["equality_residual"], 1e-6)
        self.assertLess(diag["max_kkt_margin_violation"], 1e-3)
        self.assertLess(diag["max_complementary_slackness_residual"], 1e-3)

        qp_predictions = predict_linear_svm_qp(result, X)
        library = SVC(kernel="linear", C=1.0, class_weight="balanced", tol=1e-8)
        library.fit(X, y)
        library_predictions = library.predict(X)
        self.assertTrue(set(np.unique(qp_predictions)).issubset({-1.0, 1.0}))
        np.testing.assert_array_equal(qp_predictions, library_predictions)

    def test_bias_recovery_without_free_support_vectors(self):
        X = np.array([[-0.02], [0.02], [-0.01], [0.01]])
        y = np.array([-1.0, 1.0, -1.0, 1.0])
        result = solve_linear_svm_qp(X, y, C=0.1)
        self.assertEqual(result["diagnostics"]["bias_method"], "kkt_interval_midpoint")
        self.assertTrue(np.isfinite(result["bias"]))
        self.assertAlmostEqual(float(y @ result["alpha"]), 0.0, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
