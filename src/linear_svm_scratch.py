"""A small linear SVM trained from scratch with mini-batch hinge-loss subgradient descent."""
import numpy as np
from src.class_weights import class_penalty_weights

class LinearSVMScratch:
    """Educational primal linear SVM; labels must be encoded as -1 and +1."""
    def __init__(self, C=1.0, iterations=1_000, learning_rate=0.05, batch_size=64,
                 random_state=42, class_weight=None):
        if C <= 0:
            raise ValueError("C must be positive.")
        self.C = C
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state
        self.class_weight = class_weight
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def fit(self, X, y):
        """Minimize lambda/2*||w||^2 + mean(weight_i*max(0, 1-y*f(x))).

        This is the usual soft-margin objective divided by C*n, with
        lambda=1/(C*n). Optional balanced class weights multiply hinge terms;
        the intercept is not regularized.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if set(np.unique(y)) != {-1.0, 1.0}:
            raise ValueError("Scratch SVM labels must contain both -1 and +1.")
        row_count, feature_count = X.shape
        sample_weights = class_penalty_weights(y, self.class_weight)
        regularization = 1.0 / (self.C * row_count)
        self.weights = np.zeros(feature_count, dtype=float)
        self.bias = 0.0
        self.loss_history = []
        rng = np.random.default_rng(self.random_state)

        batches_per_pass = max(1, int(np.ceil(row_count / self.batch_size)))
        order = rng.permutation(row_count)
        cursor = 0
        for iteration in range(self.iterations):
            # Shuffle after one pass through the training rows.
            if cursor >= row_count:
                order = rng.permutation(row_count)
                cursor = 0
            batch = order[cursor:min(cursor + self.batch_size, row_count)]
            cursor += len(batch)
            batch_X = X[batch]
            batch_y = y[batch]
            margins = batch_y * (batch_X @ self.weights + self.bias)
            violating = margins < 1.0
            gradient_w = regularization * self.weights
            gradient_b = 0.0
            if np.any(violating):
                # Divide by the whole batch to match the mean hinge loss.
                weighted_y = batch_y[violating] * sample_weights[batch][violating]
                gradient_w -= np.sum(weighted_y[:, None] * batch_X[violating], axis=0) / len(batch)
                gradient_b = -float(np.sum(weighted_y) / len(batch))
            # Reduce the step gradually after each approximate training pass.
            step_size = self.learning_rate / np.sqrt(1.0 + iteration / batches_per_pass)
            self.weights -= step_size * gradient_w
            self.bias -= step_size * gradient_b
            # Record the objective after every weight update, as an actual
            # convergence trace rather than only one value per epoch.
            margins = y * (X @ self.weights + self.bias)
            hinge_loss = np.mean(sample_weights * np.maximum(0.0, 1.0 - margins))
            objective = 0.5 * regularization * np.dot(self.weights, self.weights) + hinge_loss
            self.loss_history.append(float(objective))
        return self

    def decision_function(self, X):
        """Return signed distances (before normalization by ||w||)."""
        if self.weights is None:
            raise RuntimeError("Call fit() before decision_function().")
        return np.asarray(X, dtype=float) @ self.weights + self.bias

    def predict(self, X):
        """Return labels in the -1/+1 convention."""
        return np.where(self.decision_function(X) >= 0.0, 1, -1)
