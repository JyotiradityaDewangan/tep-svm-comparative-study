"""Explain soft-margin SVM convex optimization and optionally solve a tiny QP."""
import numpy as np
from src.class_weights import class_penalty_weights

def solve_linear_svm_qp(X, y, C=1.0, class_weight=None, tolerance=1e-6):
    """Solve a linear soft-margin SVM dual problem with CVXOPT.

    Inputs are small, scaled feature arrays and labels encoded as -1/+1.
    CVXOPT solves min 1/2*a'P*a - 1'a, with y'a=0 and alpha bounds C_i.
    C_i is C for unweighted fits, or C times the balanced class penalty.
    """
    try:
        from cvxopt import matrix, solvers
    except ImportError as exc:
        raise ImportError("Install CVXOPT to solve the educational SVM QP.") from exc
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(X) != len(y) or set(np.unique(y)) != {-1.0, 1.0}:
        raise ValueError("X and y must match, and y must contain both -1 and +1.")
    if C <= 0:
        raise ValueError("C must be positive.")
    row_count = len(y)
    upper_bounds = C * class_penalty_weights(y, class_weight)
    signed_gram = np.outer(y, y) * (X @ X.T)
    P = matrix(0.5 * (signed_gram + signed_gram.T), tc="d")
    q = matrix(-np.ones(row_count), tc="d")
    # The first block expresses -alpha <= 0; the second alpha <= C.
    G = matrix(np.vstack((-np.eye(row_count), np.eye(row_count))), tc="d")
    h = matrix(np.concatenate((np.zeros(row_count), upper_bounds)), tc="d")
    A = matrix(y.reshape(1, -1), tc="d")
    b_equal = matrix(np.zeros(1), tc="d")
    old_options = dict(solvers.options)
    solvers.options["show_progress"] = False
    solvers.options["abstol"] = 1e-9
    solvers.options["reltol"] = 1e-9
    solvers.options["feastol"] = 1e-9
    solvers.options["maxiters"] = 1_000
    try:
        solution = solvers.qp(P, q, G, h, A, b_equal)
    finally:
        solvers.options.clear()
        solvers.options.update(old_options)
    if solution["status"] != "optimal":
        raise RuntimeError(f"CVXOPT did not find an optimal SVM QP solution: {solution['status']}.")
    alpha = np.asarray(solution["x"]).reshape(-1)
    weights = X.T @ (alpha * y)
    support = alpha > tolerance
    free_support = support & (alpha < upper_bounds - tolerance)
    if not np.any(support):
        raise RuntimeError("CVXOPT returned no support vectors; cannot derive intercept.")
    scores_without_bias = X @ weights
    if np.any(free_support):
        bias = float(np.mean(y[free_support] - scores_without_bias[free_support]))
        bias_method = "free_support_vectors"
    else:
        # KKT conditions give a feasible interval for b even when every alpha
        # is at a box bound. Prefer its midpoint over averaging bound vectors.
        lower_candidates = []
        upper_candidates = []
        at_zero = alpha <= tolerance
        at_upper = alpha >= upper_bounds - tolerance
        lower_candidates.extend((1.0 - scores_without_bias[at_zero & (y == 1)]).tolist())
        upper_candidates.extend((-1.0 - scores_without_bias[at_zero & (y == -1)]).tolist())
        lower_candidates.extend((-1.0 - scores_without_bias[at_upper & (y == -1)]).tolist())
        upper_candidates.extend((1.0 - scores_without_bias[at_upper & (y == 1)]).tolist())
        lower = max(lower_candidates) if lower_candidates else -np.inf
        upper = min(upper_candidates) if upper_candidates else np.inf
        if np.isfinite(lower) and np.isfinite(upper) and lower <= upper + tolerance:
            bias = float((lower + upper) / 2.0)
            bias_method = "kkt_interval_midpoint"
        else:
            # Numerical solver tolerance can make the interval slightly empty.
            # Keep a visible fallback indicator so callers can inspect it.
            bias = float(np.mean(y[support] - scores_without_bias[support]))
            bias_method = "support_vector_average_fallback"

    margins = y * (scores_without_bias + bias)
    hinge = np.maximum(0.0, 1.0 - margins)
    primal_objective = float(0.5 * np.dot(weights, weights) + np.dot(upper_bounds, hinge))
    dual_objective = float(np.sum(alpha) - 0.5 * alpha @ signed_gram @ alpha)
    duality_gap = primal_objective - dual_objective
    at_lower = alpha <= tolerance
    at_upper = alpha >= upper_bounds - tolerance
    free = ~(at_lower | at_upper)
    kkt_margin_violations = []
    if np.any(at_lower):
        kkt_margin_violations.extend(np.maximum(1.0 - margins[at_lower], 0.0).tolist())
    if np.any(at_upper):
        kkt_margin_violations.extend(np.maximum(margins[at_upper] - 1.0, 0.0).tolist())
    if np.any(free):
        kkt_margin_violations.extend(np.abs(margins[free] - 1.0).tolist())
    diagnostics = {
        "max_alpha_bound_violation": float(max(0.0, -np.min(alpha), np.max(alpha - upper_bounds))),
        "equality_residual": float(abs(y @ alpha)),
        "stationarity_residual": float(np.linalg.norm(weights - X.T @ (alpha * y))),
        "max_kkt_margin_violation": float(max(kkt_margin_violations, default=0.0)),
        "max_complementary_slackness_residual": float(np.max(np.abs((upper_bounds - alpha) * hinge))),
        "primal_objective": primal_objective,
        "dual_objective": dual_objective,
        "duality_gap": float(duality_gap),
        "bias_method": bias_method,
        "tolerance": tolerance,
    }
    return {"weights": weights, "bias": bias, "alpha": alpha,
            "support_vector_count": int(np.sum(support)), "status": solution["status"],
            "upper_bounds": upper_bounds, "diagnostics": diagnostics}


def predict_linear_svm_qp(model, X):
    """Predict signed (-1/+1) labels from a solved linear QP model."""
    scores = np.asarray(X, dtype=float) @ model["weights"] + model["bias"]
    return np.where(scores >= 0.0, 1.0, -1.0)

def explain_formulation():
    print("\nEducational note: soft-margin SVM minimizes 1/2||w||^2 + C*sum(xi_i).")
    print("Constraints: y_i(w^T x_i + b) >= 1-xi_i and xi_i >= 0.")
    print("The squared norm is convex; linear constraints make a convex quadratic program.")
    print("Lagrange multipliers weight constraints; KKT conditions characterize an optimum.")
    print("Support vectors are observations with nonzero influence on the fitted boundary.")
    try:
        import cvxopt  # Check dependency early and keep this note beginner-friendly.
    except ImportError:
        print("CVXOPT is not installed; the optional small QP example was skipped. Install cvxopt to run it.")
        return None
    # Two one-dimensional groups: x=-2,-1 (class -1), x=1,2 (class +1).
    # QP dual: min 1/2 alpha^T P alpha - 1^T alpha, subject to y^T alpha=0, alpha>=0.
    X=np.array([[-2.],[-1.],[1.],[2.]])
    y=np.array([-1.,-1.,1.,1.])
    result=solve_linear_svm_qp(X,y,C=10.0)
    alpha=result["alpha"]; w=result["weights"]; intercept=result["bias"]
    print("Tiny CVXOPT QP solved successfully.")
    print(f"P is the label-weighted Gram matrix ({len(y)}x{len(y)}); q=-1; G and h encode 0<=alpha<=C.")
    print(f"A=y^T and b=0 encode the equality y^T alpha=0. w={w.tolist()}, intercept={intercept:.4f}.")
    print("The separating boundary is w^T x + b = 0; for this example it lies at x=0.")
    return {"alpha":alpha.tolist(),"w":w.tolist(),"intercept":intercept}
