"""Write per-run manifests and reports from measured experiment objects."""
from datetime import datetime, timezone
import json
import platform
import sys
import importlib.util

import numpy as np
import pandas as pd
import sklearn

import config


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if pd.isna(value):
        return None
    return str(value)


def _frame_records(frame):
    if not isinstance(frame, pd.DataFrame):
        return frame
    return [{str(key): _json_value(value) for key, value in row.items()}
            for row in frame.to_dict(orient="records")]


def _config_snapshot():
    keys = (
        "SAMPLE_SIZE", "NORMAL_LABEL", "TEST_SIZE", "RANDOM_STATE", "SVM_KERNEL",
        "SVM_C", "SVM_GAMMA", "NORMALIZE_DATA", "USE_CLASS_BALANCING",
        "RUN_SVM_EXPERIMENTS", "RUN_PCA_EXPERIMENT", "PCA_EXPLAINED_VARIANCE_TARGET",
        "EXPERIMENT_KERNELS", "EXPERIMENT_C_VALUES", "EXPERIMENT_MAX_ROWS",
        "EXPERIMENT_SAMPLE_SIZES", "SCRATCH_COMPARE_ROWS", "SCRATCH_ITERATIONS",
        "SCRATCH_LEARNING_RATE", "QP_MAX_ROWS", "MAX_SVC_ROWS",
    )
    return {key.lower(): getattr(config, key) for key in keys}


def _table_markdown(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return "No result rows were produced."
    view = frame.copy()
    if view.index.name and view.index.name not in view.columns:
        view = view.reset_index()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    headers = [str(column) for column in view.columns]
    rows = [[str(value) for value in row] for row in view.itertuples(index=False, name=None)]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    head = "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(headers)) + " |"
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |" for row in rows]
    return "\n".join([head, divider, *body])


def write_run_records(run_id, dataset_summary, prepared, main_metrics=None,
                      experiments=None, pca_result=None, run_kind="full",
                      confusion_matrix=None, input_hashes=None):
    """Persist a machine-readable manifest and a report sourced from this run."""
    run_dir = config.RESULTS_DIR
    run_dir.mkdir(parents=True, exist_ok=True)
    experiment_tables = {}
    experiment_metadata = {}
    if experiments:
        experiment_metadata = experiments.get("_run_metadata", {})
        experiment_tables = {key: value for key, value in experiments.items()
                             if not key.startswith("_")}
    if pca_result is not None:
        experiment_tables["pca"] = pca_result

    cvxopt_version = "not installed"
    if importlib.util.find_spec("cvxopt") is not None:
        import cvxopt
        cvxopt_version = getattr(cvxopt, "__version__", "installed; version unavailable")
    manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_kind": run_kind,
        "dataset": {
            "name": dataset_summary.get("dataset_name"),
            "source_filenames": dataset_summary.get("source_files", []),
            "sampled_rows": dataset_summary.get("sampled_observations"),
            "rows_scanned": dataset_summary.get("rows_scanned"),
            "source_rows": dataset_summary.get("rows_by_source_file", {}),
            "sampled_class_distribution": dataset_summary.get("sampled_class_distribution_by_source", {}),
            "label_mapping": {"NORMAL": config.NORMAL_LABEL, "FAULTY": "all observed non-normal labels"},
            "feature_names_in_order": dataset_summary.get("input_features", []),
            "metadata_columns_excluded": dataset_summary.get("metadata_columns", []) + dataset_summary.get("internal_columns_excluded", []),
        },
        "split": {
            "main_strategy": prepared.get("split_strategy"),
            "training_rows": int(len(prepared.get("y_train", []))),
            "testing_rows": int(len(prepared.get("y_test", []))),
            "training_class_distribution": {"NORMAL": int(np.sum(prepared.get("y_train", []) == 0)),
                                             "FAULTY": int(np.sum(prepared.get("y_train", []) == 1))},
            "testing_class_distribution": {"NORMAL": int(np.sum(prepared.get("y_test", []) == 0)),
                                            "FAULTY": int(np.sum(prepared.get("y_test", []) == 1))},
            "experiment_inner_split": experiment_metadata,
        },
        "preprocessing": {
            "imputer": "median; fit on each model's fitting observations only",
            "scaler": "StandardScaler; fit on each model's fitting observations only" if config.NORMALIZE_DATA else "disabled",
            "pca": {"explained_variance_target": config.PCA_EXPLAINED_VARIANCE_TARGET,
                    "fit_scope": "training observations only"},
        },
        "model_configuration": {
            "main_and_pca_classifier": {"kernel": config.SVM_KERNEL, "C": config.SVM_C,
                                        "gamma": config.SVM_GAMMA,
                                        "class_weight": "balanced" if config.USE_CLASS_BALANCING else None},
            "class_weighting": "balanced" if config.USE_CLASS_BALANCING else None,
            "experiment_settings": _config_snapshot(),
        },
        "software": {"python": platform.python_version(), "platform": platform.platform(),
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__, "cvxopt": cvxopt_version},
        "metrics": {
            "main_model": ({**_json_value(main_metrics),
                            "confusion_matrix": _json_value(confusion_matrix)}
                           if main_metrics is not None else None),
            "experiments": {key: _frame_records(frame) for key, frame in experiment_tables.items()},
        },
        "input_file_hashes": input_hashes or {},
        "artifacts": {"relative_run_folder": f"results/runs/{run_id}",
                      "main_model_saved": bool(main_metrics is not None and
                          (run_dir / "models" / "trained_svm_model.pkl").is_file())},
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_value), encoding="utf-8")

    lines = [
        f"# Experiment run {run_id}", "",
        f"Generated: {manifest['timestamp_utc']} (UTC)",
        f"Run type: {run_kind}",
        "",
        "Training times measure estimator fitting only and vary with machine load and software environment.",
        "This report is generated from the measured result objects for this run.",
        "",
    ]
    if main_metrics is not None:
        lines += ["## Main RBF SVM", "", _table_markdown(pd.DataFrame([main_metrics])), ""]
    for name, frame in experiment_tables.items():
        lines += [f"## {name.replace('_', ' ').title()}", "", _table_markdown(frame), ""]
    report_path = run_dir / "experiment_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return manifest_path, report_path
