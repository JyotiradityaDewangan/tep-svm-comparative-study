import unittest

import numpy as np

from src.class_weights import class_penalty_weights
from src.linear_svm_scratch import LinearSVMScratch


class ScratchSVMTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[-2.0], [-1.0], [-0.2], [0.2], [1.0], [2.0]])
        self.y = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0])

    def test_training_predictions_finite_and_objective_matches_manual_calculation(self):
        model = LinearSVMScratch(C=1.0, iterations=60, learning_rate=0.02,
                                 batch_size=3, random_state=3)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        self.assertTrue(set(np.unique(predictions)).issubset({-1.0, 1.0}))
        self.assertTrue(np.isfinite(model.weights).all())
        self.assertTrue(np.isfinite(model.bias))
        self.assertEqual(len(model.loss_history), 60)

        margins = self.y * (self.X @ model.weights + model.bias)
        manual_hinge = np.maximum(0.0, 1.0 - margins).mean()
        manual_regularizer = 0.5 * np.dot(model.weights, model.weights) / (1.0 * len(self.y))
        self.assertAlmostEqual(model.loss_history[-1], manual_regularizer + manual_hinge, places=12)

    def test_balanced_class_weights_match_expected_multipliers(self):
        labels = np.array([-1, -1, -1, -1, 1, 1])
        weights = class_penalty_weights(labels, "balanced")
        np.testing.assert_allclose(weights, [0.75, 0.75, 0.75, 0.75, 1.5, 1.5])
        model = LinearSVMScratch(C=1.0, iterations=15, learning_rate=0.01,
                                 batch_size=6, random_state=2,
                                 class_weight="balanced").fit(self.X, self.y)
        self.assertEqual(len(model.loss_history), 15)
        self.assertTrue(np.isfinite(model.loss_history).all())
        weights = class_penalty_weights(self.y, "balanced")
        margins = self.y * (self.X @ model.weights + model.bias)
        manual = (0.5 * np.dot(model.weights, model.weights) / len(self.y)
                  + np.mean(weights * np.maximum(0.0, 1.0 - margins)))
        self.assertAlmostEqual(model.loss_history[-1], manual, places=12)

    def test_different_c_values_change_regularized_solution(self):
        low_c = LinearSVMScratch(C=0.05, iterations=200, learning_rate=0.01,
                                 batch_size=6, random_state=8).fit(self.X, self.y)
        high_c = LinearSVMScratch(C=20.0, iterations=200, learning_rate=0.01,
                                  batch_size=6, random_state=8).fit(self.X, self.y)
        self.assertFalse(np.allclose(low_c.weights, high_c.weights))
        self.assertTrue(np.isfinite(low_c.loss_history).all())
        self.assertTrue(np.isfinite(high_c.loss_history).all())


if __name__ == "__main__":
    unittest.main()
