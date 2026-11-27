"""Evaluate saved training artifacts on the official, unsampled TEP test CSVs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.metrics.pairwise import rbf_kernel

import config
from src.data_loader import _target_column
from src.svm_model import load_artifacts
from src.input_hashes import collect_training_hashes

OFFICIAL_FILES = ("TEP_FaultFree_Testing.csv", "TEP_Faulty_Testing.csv")
CHUNK_ROWS = 50_000
KERNEL_ROWS = 1_024


def _sha256(path):
    """Hash large CSV files without reading them into memory at once."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _class_from_file_name(path):
    name = path.name.lower()
    if "faultfree" in name or "normal" in name:
        return 0
    if "faulty" in name or "fault_" in name:
        return 1
    return None


def _labels_for_chunk(frame, path):
    target = _target_column(frame.columns)
    if target is not None:
        values = pd.to_numeric(frame[target], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Target column '{target}' contains missing or non-numeric labels in {path.name}.")
        return (values != config.NORMAL_LABEL).astype("int8").to_numpy(), target
    file_label = _class_from_file_name(path)
    if file_label is None:
        raise ValueError(f"Cannot determine NORMAL/FAULTY labels for {path.name}: no target column or recognized filename.")
    return np.full(len(frame), file_label, dtype="int8"), None


def _score_saved_rbf(model, X, kernel_rows=KERNEL_ROWS):
    """Evaluate a fitted binary RBF SVC in bounded, BLAS-friendly batches.

    This is the exact fitted kernel SVM decision equation. It avoids libsvm's
    slow per-observation prediction path on millions of official test rows.
    No model parameter is fitted or changed here.
    """
    if (getattr(model, "kernel", None) != "rbf" or len(model.classes_) != 2
            or not np.array_equal(model.classes_, np.array([0, 1]))):
        raise ValueError("The saved model must be a fitted binary RBF SVC with classes 0 and 1.")
    X = np.asarray(X, dtype=float)
    support_vectors = np.asarray(model.support_vectors_, dtype=float)
    coefficients = np.asarray(model.dual_coef_[0], dtype=float)
    scores = np.empty(len(X), dtype=float)
    for start in range(0, len(X), kernel_rows):
        stop = min(start + kernel_rows, len(X))
        kernel_values = rbf_kernel(X[start:stop], support_vectors, gamma=model._gamma)
        scores[start:stop] = kernel_values @ coefficients + float(model.intercept_[0])
    return scores


def evaluate_official_files(model_dir, testing_dir=None, output_dir=None,
                            chunk_rows=CHUNK_ROWS, run_id=None, training_dir=None):
    """Score every row from both official files using saved fitted objects only."""
    testing_dir = Path(testing_dir or (config.ROOT_DIR / "data" / "testing"))
    paths = [testing_dir / name for name in OFFICIAL_FILES]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Official testing CSV file(s) missing under {testing_dir}: {', '.join(missing)}"
        )
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer.")

    model, imputer, scaler, features = load_artifacts(model_dir)
    if len(features) != 52:
        raise ValueError(f"Saved model expects {len(features)} features; official TEP evaluation requires exactly 52.")
    if getattr(model, "kernel", None) != "rbf":
        raise ValueError("Official testing requires a saved RBF SVM model.")

    y_all, prediction_all, score_all = [], [], []
    file_details = {}
    started = time.perf_counter()
    for path in paths:
        print(f"Reading and scoring every observation in {path.name}...", flush=True)
        rows = 0
        normal = faulty = 0
        target_name = None
        reader = None
        try:
            reader = pd.read_csv(path, chunksize=chunk_rows, low_memory=False)
            for frame in reader:
                missing_features = [name for name in features if name not in frame.columns]
                if missing_features:
                    raise ValueError(f"{path.name} is missing saved model features: {missing_features}")
                labels, found_target = _labels_for_chunk(frame, path)
                if target_name is None:
                    target_name = found_target
                elif found_target != target_name:
                    raise ValueError(f"Target column changed while reading {path.name}.")
                try:
                    sensor_values = frame.loc[:, features].apply(pd.to_numeric, errors="raise")
                    # Keep feature names for a DataFrame-fitted imputer while
                    # preserving the saved model's exact column order.
                    transformed = imputer.transform(sensor_values)
                    if scaler is not None:
                        transformed = scaler.transform(transformed)
                except Exception as exc:
                    raise ValueError(f"Could not preprocess sensor columns in {path.name}: {exc}") from exc
                scores = _score_saved_rbf(model, transformed)
                # For binary SVC, positive decision values select classes_[1].
                predictions = np.where(scores >= 0.0, model.classes_[1], model.classes_[0]).astype("int8")
                if len(predictions) != len(labels) or not np.isfinite(scores).all():
                    raise ValueError(f"Model returned invalid outputs for rows in {path.name}.")
                y_all.append(labels)
                prediction_all.append(predictions)
                score_all.append(scores)
                rows += len(labels)
                normal += int(np.sum(labels == 0))
                faulty += int(np.sum(labels == 1))
                if rows % 500_000 == 0:
                    print(f"  Evaluated {rows:,} rows from {path.name}.", flush=True)
        except Exception as exc:
            if isinstance(exc, (FileNotFoundError, ValueError)):
                raise
            raise ValueError(f"Could not read official testing CSV '{path.name}': {exc}") from exc
        finally:
            if reader is not None:
                reader.close()
        if rows == 0:
            raise ValueError(f"Official testing CSV contains no observations: {path.name}")
        file_details[path.name] = {
            "observations": rows,
            "class_distribution": {"NORMAL": normal, "FAULTY": faulty},
            "target_column": target_name,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }

    y_true = np.concatenate(y_all)
    predictions = np.concatenate(prediction_all)
    scores = np.concatenate(score_all)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    metrics = {
        "observations": int(len(y_true)),
        "feature_count": len(features),
        "normal_observations": int(np.sum(y_true == 0)),
        "faulty_observations": int(np.sum(y_true == 1)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None,
        "confusion_matrix": matrix.tolist(),
        "evaluation_seconds": time.perf_counter() - started,
    }

    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = config.RESULTS_ROOT_DIR / "official_testing" / f"evaluation_{stamp}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "official_test_metrics.csv", index=False)
    pd.DataFrame(matrix, index=["true_NORMAL", "true_FAULTY"],
                 columns=["predicted_NORMAL", "predicted_FAULTY"]).to_csv(
                     output_dir / "confusion_matrix.csv")
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix, display_labels=["NORMAL", "FAULTY"]).plot(
        ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title("Official TEP Testing Confusion Matrix")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation": "official testing; no sampling; saved model and fitted preprocessing only",
        "model_directory": str(Path(model_dir).name),
        "features_in_model_order": list(features),
        "files": file_details,
        "training_csvs": collect_training_hashes(training_dir),
        "metrics": metrics,
    }
    (output_dir / "official_test_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n========================================")
    print("OFFICIAL TEP TESTING EVALUATION")
    print("========================================")
    for filename, detail in file_details.items():
        counts = detail["class_distribution"]
        print(f"{filename}: {detail['observations']:,} observations; "
              f"NORMAL={counts['NORMAL']:,}, FAULTY={counts['FAULTY']:,}")
    print(f"Total observations evaluated: {metrics['observations']:,}")
    print(f"Class distribution: NORMAL={metrics['normal_observations']:,}, "
          f"FAULTY={metrics['faulty_observations']:,}")
    print(f"Feature count: {metrics['feature_count']} (saved model feature order)")
    for name in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        value = metrics[name]
        print(f"{name.replace('_', ' ').title()}: {'N/A' if value is None else f'{value:.6f}'}")
    print("Confusion matrix (rows=true NORMAL,FAULTY; columns=predicted NORMAL,FAULTY):")
    print(matrix)
    print(f"Evaluation time: {metrics['evaluation_seconds']:.2f} seconds")
    print(f"Saved official results to: {output_dir}")
    return metrics, output_dir


def run_official_testing(model_dir, run_dir=None):
    """Public entry point, attaching official result hashes to the saved-run context."""
    run_id = Path(run_dir).name if run_dir is not None else None
    return evaluate_official_files(model_dir, run_id=run_id)
