"""Small helper for scikit-learn-compatible balanced class penalties."""
import numpy as np


def class_penalty_weights(y, class_weight=None):
    """Return one nonnegative penalty multiplier per row.

    For ``balanced``, class k receives n / (number_of_classes * n_k), matching
    the class-weight formula used by scikit-learn's linear classifiers.
    """
    y = np.asarray(y)
    weights = np.ones(len(y), dtype=float)
    if class_weight is None:
        return weights
    if class_weight != "balanced":
        raise ValueError("class_weight must be None or 'balanced'.")
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or np.any(counts == 0):
        raise ValueError("Balanced class weights require at least two observed classes.")
    multipliers = {label: len(y) / (len(classes) * count)
                   for label, count in zip(classes, counts)}
    return np.asarray([multipliers[label] for label in y], dtype=float)
