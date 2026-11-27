"""Compare the configured RBF SVC on original and PCA sensor features."""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import config
from src.evaluation import evaluate
from src.svm_model import make_model


def _fit_and_score(X_train, y_train, X_test, y_test, feature_count):
    """Fit the same configured SVC and return metrics plus timings."""
    # PCA comparison is specifically an RBF comparison even if the project's
    # configurable primary kernel is changed for another study.
    model = make_model(kernel="rbf")
    started = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - started
    metrics, _, _, _, _ = evaluate(model, X_test, y_test)
    return {
        "features": feature_count,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
        "training_time_seconds": training_seconds,
        "prediction_time_seconds": metrics["testing_time_seconds"],
    }


def _save_plots(pca, component_count, original_count, rows):
    """Save the four requested PCA experiment charts."""
    folder = config.RESULTS_DIR / "pca"
    folder.mkdir(parents=True, exist_ok=True)
    component_numbers = np.arange(1, len(pca.explained_variance_ratio_) + 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(component_numbers, pca.explained_variance_ratio_)
    ax.set(title="Individual Explained Variance", xlabel="Principal component number",
           ylabel="Explained variance ratio")
    ax.set_xticks(component_numbers)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / "explained_variance.png", dpi=160)
    plt.close(fig)

    cumulative = np.cumsum(pca.explained_variance_ratio_) * 100
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(component_numbers, cumulative, marker="o", markersize=3, label="Cumulative variance")
    ax.axhline(config.PCA_EXPLAINED_VARIANCE_TARGET * 100, color="darkorange",
               linestyle="--", label=f"Target ({config.PCA_EXPLAINED_VARIANCE_TARGET:.0%})")
    ax.axvline(component_count, color="seagreen", linestyle=":",
               label=f"Selected components ({component_count})")
    ax.set(title="Cumulative Explained Variance", xlabel="Number of principal components",
           ylabel="Cumulative explained variance (%)", ylim=(0, 102))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / "cumulative_explained_variance.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Original sensor features", "Retained PCA components"]
    values = [original_count, component_count]
    bars = ax.bar(labels, values, color=["steelblue", "darkorange"])
    ax.bar_label(bars, padding=3)
    ax.set(title="Original Features vs PCA Features", ylabel="Number of features")
    ax.set_ylim(0, max(values) * 1.15)
    fig.tight_layout()
    fig.savefig(folder / "dimensionality_reduction.png", dpi=160)
    plt.close(fig)

    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    positions = np.arange(len(metric_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(positions - width / 2, [rows[0][name] for name in metric_names],
           width, label="Original features")
    ax.bar(positions + width / 2, [rows[1][name] for name in metric_names],
           width, label="PCA features")
    ax.set(title="RBF SVM Performance: Original vs PCA", ylabel="Score", ylim=(0, 1.05))
    ax.set_xticks(positions, metric_labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / "pca_svm_performance_comparison.png", dpi=160)
    plt.close(fig)


def fit_pca_training_split(X_train, X_test, explained_variance_target=0.95):
    """Fit PCA on training features only and return transformed matrices."""
    pca = PCA(n_components=explained_variance_target, svd_solver="full")
    X_train_pca = pca.fit_transform(np.asarray(X_train))
    X_test_pca = pca.transform(np.asarray(X_test))
    retained = float(pca.explained_variance_ratio_.sum())
    if retained + 1e-12 < explained_variance_target:
        raise RuntimeError("PCA did not retain the configured explained variance target.")
    return pca, X_train_pca, X_test_pca


def run_pca_experiment(X_train, y_train, X_test, y_test):
    """Run PCA using train-fitted data from the existing shared split."""
    print("\n========================================")
    print("PCA EXPERIMENT")
    print("========================================")
    print("[1/5] Loading existing TEP dataset... (loaded by the shared project workflow)")
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    original_count = X_train.shape[1]
    print("\n[2/5] Preparing original sensor features...")
    print(f"Original number of features: {original_count}")
    print(f"Training observations: {len(y_train):,}")
    print(f"Testing observations: {len(y_test):,}")
    print("Metadata and labels are already excluded by the shared data loader.")
    print("The shared median imputer and scaler were fit using training rows only.")

    print("\n[3/5] Applying PCA...")
    target = config.PCA_EXPLAINED_VARIANCE_TARGET
    print(f"Target explained variance: {target:.0%}")
    # PCA is fitted only on the already scaled training observations. Scikit-learn
    # selects the smallest component count that reaches the requested variance.
    pca, X_train_pca, X_test_pca = fit_pca_training_split(X_train, X_test, target)
    component_count = X_train_pca.shape[1]
    variance_retained = float(pca.explained_variance_ratio_.sum())
    reduction = (original_count - component_count) / original_count * 100
    print(f"Number of retained components: {component_count}")
    print(f"Actual explained variance retained: {variance_retained:.2%}")
    print(f"Dimensionality reduction: {reduction:.2f}%")
    print("Principal components are new combinations of the original sensor features.")

    print("\n[4/5] Training SVM with original features...")
    original = _fit_and_score(X_train, y_train, X_test, y_test, original_count)
    _print_metrics(original)

    print("\n[5/5] Training SVM with PCA features...")
    transformed = _fit_and_score(X_train_pca, y_train, X_test_pca, y_test, component_count)
    _print_metrics(transformed)

    rows = [original, transformed]
    table = pd.DataFrame(rows, index=["Original features", "PCA features"])
    table.index.name = "approach"
    table.insert(0, "explained_variance_target", [np.nan, target])
    table.insert(1, "explained_variance_retained", [np.nan, variance_retained])
    folder = config.RESULTS_DIR / "pca"
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "pca_svm_comparison.csv"
    table.reset_index().to_csv(csv_path, index=False)
    if config.SAVE_EXCEL_REPORTS:
        with pd.ExcelWriter(folder / "pca_svm_comparison.xlsx", engine="openpyxl") as writer:
            table.reset_index().to_excel(writer, sheet_name="PCA SVM Comparison", index=False)
    _save_plots(pca, component_count, original_count, rows)

    print("\n========================================")
    print("PCA COMPARISON SUMMARY")
    print("========================================")
    print(f"{'Metric':<24}{'Original Features':>21}{'PCA Features':>18}")
    print("-" * 63)
    metrics = [
        ("Features", original_count, component_count),
        ("Accuracy", original["accuracy"], transformed["accuracy"]),
        ("Precision", original["precision"], transformed["precision"]),
        ("Recall", original["recall"], transformed["recall"]),
        ("F1-score", original["f1_score"], transformed["f1_score"]),
        ("ROC-AUC", original["roc_auc"], transformed["roc_auc"]),
        ("Training Time (s)", original["training_time_seconds"], transformed["training_time_seconds"]),
        ("Prediction Time (s)", original["prediction_time_seconds"], transformed["prediction_time_seconds"]),
    ]
    for name, first, second in metrics:
        if name == "Features":
            print(f"{name:<24}{first:>21}{second:>18}")
        else:
            print(f"{name:<24}{first:>21.6f}{second:>18.6f}")
    print("\n========================================")
    print("PCA EXPERIMENT COMPLETED")
    print("========================================")
    print(f"CSV: {csv_path}")
    if config.SAVE_EXCEL_REPORTS:
        print(f"Excel: {folder / 'pca_svm_comparison.xlsx'}")
    for name in ("explained_variance.png", "cumulative_explained_variance.png",
                 "dimensionality_reduction.png", "pca_svm_performance_comparison.png"):
        print(f"Plot: {folder / name}")
    return table


def _print_metrics(metrics):
    for name in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        print(f"{name.replace('_', ' ').title()}: {metrics[name]:.6f}")
    print(f"Training time: {metrics['training_time_seconds']:.3f} seconds")
    print(f"Prediction time: {metrics['prediction_time_seconds']:.3f} seconds")
