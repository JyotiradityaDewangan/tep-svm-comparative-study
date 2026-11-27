"""Create collision-free output locations while preserving historical files."""
from datetime import datetime, timezone
import json
import uuid

import config


def create_run_id():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def configure_run(run_id=None):
    run_id = run_id or create_run_id()
    run_dir = config.RESULTS_ROOT_DIR / "runs" / run_id
    config.RUN_ID = run_id
    config.RESULTS_DIR = run_dir
    config.PLOTS_DIR = run_dir / "plots"
    config.HEATMAPS_DIR = run_dir / "heatmaps"
    config.EXCEL_DIR = run_dir / "excel_reports"
    config.CSV_DIR = run_dir / "csv_reports"
    config.PREDICTIONS_DIR = run_dir / "predictions"
    config.SUMMARY_DIR = run_dir / "summary"
    config.LOG_DIR = run_dir / "logs"
    config.MODEL_DIR = run_dir / "models"
    config.PROCESSED_DIR = run_dir / "data" / "processed"
    return run_id, run_dir


def latest_saved_model_dir():
    """Return the latest completed run's model folder or legacy models folder."""
    run_root = config.RESULTS_ROOT_DIR / "runs"
    if run_root.exists():
        for run_dir in sorted(run_root.iterdir(), reverse=True):
            manifest_path = run_dir / "run_manifest.json"
            model_path = run_dir / "models" / "trained_svm_model.pkl"
            if not (manifest_path.is_file() and model_path.is_file()):
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("artifacts", {}).get("main_model_saved"):
                return run_dir / "models", run_dir
    return config.LEGACY_MODEL_DIR, None
