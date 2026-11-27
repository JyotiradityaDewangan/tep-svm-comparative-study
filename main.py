"""Terminal entry point for run-isolated SVM training, experiments, and prediction."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import config
from src.data_loader import load_dataset
from src.preprocessing import prepare_data
from src.svm_model import make_model, save_artifacts
from src.evaluation import evaluate, save_csv_reports, save_excel_reports
from src.visualization import create_plots
from src.optimization import explain_formulation
from src.prediction import predict_file
from src.experiments import run_svm_experiments
from src.pca_experiment import run_pca_experiment
from src.run_paths import configure_run, latest_saved_model_dir
from src.run_reporting import write_run_records
from src.official_testing import run_official_testing
from src.input_hashes import collect_input_hashes


def ensure_directories():
    experiment_dirs = [config.RESULTS_DIR / name for name in (
        "main_model", "svm_from_scratch", "cvxopt_svm", "svm_comparison",
        "c_parameter_experiment", "kernel_experiment", "convergence", "pca")]
    for path in (config.RAW_DIR, config.PROCESSED_DIR, config.MODEL_DIR,
                 config.PLOTS_DIR, config.HEATMAPS_DIR, config.EXCEL_DIR,
                 config.CSV_DIR, config.PREDICTIONS_DIR, config.SUMMARY_DIR,
                 config.LOG_DIR, *experiment_dirs):
        path.mkdir(parents=True, exist_ok=True)


def copy_main_results_into_project_folder():
    """Keep a convenient main-model bundle inside this run's own folder."""
    target = config.RESULTS_DIR / "main_model"
    source_folders = {
        "plots": config.PLOTS_DIR, "heatmaps": config.HEATMAPS_DIR,
        "excel_reports": config.EXCEL_DIR, "csv_reports": config.CSV_DIR,
        "predictions": config.PREDICTIONS_DIR, "summary": config.SUMMARY_DIR,
        "models": config.MODEL_DIR,
    }
    for name, source in source_folders.items():
        destination = target / name
        destination.mkdir(parents=True, exist_ok=True)
        if source.resolve() == destination.resolve():
            continue
        for item in source.iterdir():
            if item.is_file():
                shutil.copy2(item, destination / item.name)


def run_self_test():
    """Run the deterministic, dataset-independent synthetic software suite."""
    import unittest

    test_folder = config.ROOT_DIR / "tests"
    if not test_folder.is_dir():
        print("SELF-TEST FAILED: tests/ was not found.")
        return 1
    suite = unittest.defaultTestLoader.discover(str(test_folder), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    skipped = len(result.skipped)
    failed = len(result.failures) + len(result.errors)
    passed = result.testsRun - skipped - failed
    print("\nSYNTHETIC SOFTWARE TEST SUMMARY")
    print(f"PASSED: {passed}")
    print(f"FAILED: {failed}")
    print(f"SKIPPED: {skipped}")
    print("These tests use synthetic data and do not validate results on the real TEP dataset.")
    return 1 if failed else 0


def _raw_numeric_features(data, features):
    """Return sensor columns before any fitted imputation or scaling."""
    return data[features].apply(__import__("pandas").to_numeric, errors="coerce").to_numpy()


def _predict(args, parser):
    if not args.input:
        parser.error("--predict requires --input path/to/sensor_readings.csv")
    if args.run_id:
        if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
            raise ValueError("--run-id must be a single completed-run folder name.")
        run_dir = config.RESULTS_ROOT_DIR / "runs" / args.run_id
        model_dir = run_dir / "models"
        if not (run_dir / "run_manifest.json").is_file():
            raise FileNotFoundError(f"Completed run was not found: {args.run_id}")
    else:
        model_dir, run_dir = latest_saved_model_dir()
    if not (model_dir / "trained_svm_model.pkl").is_file():
        raise FileNotFoundError(
            f"No trained SVM model exists at {model_dir}. Download the TEP training CSVs into data/raw/ "
            "and run 'python main.py' first."
        )
    input_path = Path(args.input).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"Prediction input CSV not found: {input_path.resolve()}")
    if run_dir is not None:
        config.PREDICTIONS_DIR = run_dir / "predictions"
    predict_file(input_path, args.output, model_dir=model_dir)


def _print_confusion(confusion):
    print("Confusion matrix (rows=true NORMAL,FAULTY; columns=predicted NORMAL,FAULTY):")
    print(confusion)


def _print_saved_paths(run_dir):
    print("\nRun output files:")
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(config.RESULTS_ROOT_DIR)}")


def main():
    parser = argparse.ArgumentParser(description="Binary Tennessee Eastman Process fault detection with SVM")
    parser.add_argument("--predict", action="store_true", help="predict from a sensor CSV without training")
    parser.add_argument("--official-test", action="store_true",
                        help="evaluate saved model on official TEP testing CSVs without retraining")
    parser.add_argument("--input", help="CSV path used with --predict")
    parser.add_argument("--output", help="optional prediction CSV path")
    parser.add_argument("--run-id", help="select a completed run's model for prediction or official testing")
    parser.add_argument("--self-test", action="store_true", help="run deterministic synthetic software tests")
    parser.add_argument("--experiments-only", action="store_true", help="run selected experiments without fitting the main SVC")
    parser.add_argument("--experiment", choices=["all", "implementations", "c", "kernel",
        "convergence", "sample-size", "pca"], default="all", help="select experiments to run")
    args = parser.parse_args()
    config.validate_config()
    if args.run_id and not (args.predict or args.official_test):
        parser.error("--run-id can only be used with --predict or --official-test")
    if args.predict:
        _predict(args, parser)
        return 0
    if args.official_test:
        if args.run_id:
            if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
                raise ValueError("--run-id must be a single completed-run folder name.")
            run_dir = config.RESULTS_ROOT_DIR / "runs" / args.run_id
            model_dir = run_dir / "models"
            if not (run_dir / "run_manifest.json").is_file():
                raise FileNotFoundError(f"Completed run was not found: {args.run_id}")
        else:
            model_dir, run_dir = latest_saved_model_dir()
        run_official_testing(model_dir, run_dir=run_dir)
        return 0
    if args.self_test:
        return run_self_test()

    run_id, run_dir = configure_run()
    ensure_directories()
    print("=" * 40 + "\nCHEMICAL PROCESS FAULT DETECTION\nUSING SVM AND CONVEX OPTIMIZATION\n" + "=" * 40)
    print(f"Run ID: {run_id}")
    print(f"Run type: {'experiments-only' if args.experiments_only else 'full'}")
    print("\nSTEP 1: Loading dataset")
    data, features, dataset_summary = load_dataset()
    if config.SAMPLE_SIZE is None and dataset_summary["rows_scanned"] > 2_000_000:
        raise ValueError("SAMPLE_SIZE=None exceeds the safe SVC row limit. Set SAMPLE_SIZE within MAX_SVC_ROWS.")
    if len(data) > config.MAX_SVC_ROWS:
        raise ValueError(f"Sample contains {len(data):,} rows; SVC safety cap is {config.MAX_SVC_ROWS:,}.")
    print("Sample class counts: "
          f"NORMAL={dataset_summary['normal_observations']:,}, "
          f"FAULTY={dataset_summary['faulty_observations']:,}")

    print("\nSTEP 2: Main train/test preprocessing")
    prepared = prepare_data(data, features)
    print(f"Main split: {prepared['split_strategy']}")
    print(f"Training rows: {len(prepared['y_train']):,}; testing rows: {len(prepared['y_test']):,}")
    print("Main training class counts: "
          f"NORMAL={int(np.sum(prepared['y_train'] == 0)):,}, "
          f"FAULTY={int(np.sum(prepared['y_train'] == 1)):,}")
    print("Main testing class counts: "
          f"NORMAL={int(np.sum(prepared['y_test'] == 0)):,}, "
          f"FAULTY={int(np.sum(prepared['y_test'] == 1)):,}")
    print("Main model imputer and scaler were fitted only on main training rows.")

    experiments = None
    pca_result = None
    main_metrics = None
    main_confusion = None
    plot_paths = []
    raw_X = _raw_numeric_features(data, features)
    raw_train_X = raw_X[np.asarray(prepared["train_index"], dtype=int)]
    y_train = prepared["y_train"]
    groups = prepared["train_groups"]

    if args.experiments_only:
        print("\nMain classifier fitting skipped.")
        if args.experiment == "pca":
            if config.RUN_PCA_EXPERIMENT:
                pca_result = run_pca_experiment(prepared["X_train"], y_train,
                                                prepared["X_test"], prepared["y_test"])
            else:
                print("SKIPPED: RUN_PCA_EXPERIMENT is disabled in config.py")
        else:
            if config.RUN_SVM_EXPERIMENTS:
                experiments = run_svm_experiments(raw_train_X, y_train, groups, selected=args.experiment)
            else:
                print("SKIPPED: RUN_SVM_EXPERIMENTS is disabled in config.py")
            if args.experiment == "all" and config.RUN_PCA_EXPERIMENT:
                pca_result = run_pca_experiment(prepared["X_train"], y_train,
                                                prepared["X_test"], prepared["y_test"])
    else:
        print("\nSTEP 3: Training main SVM")
        model = make_model()
        print(f"Kernel: {config.SVM_KERNEL}; C: {config.SVM_C}; gamma: {config.SVM_GAMMA}; "
              f"class_weight: {'balanced' if config.USE_CLASS_BALANCING else 'none'}")
        started = time.perf_counter()
        model.fit(prepared["X_train"], y_train)
        main_training_seconds = time.perf_counter() - started
        print(f"Training completed in {main_training_seconds:.3f} seconds.")

        print("\nSTEP 4: Evaluating main SVM")
        main_metrics, predicted, scores, class_report, confusion = evaluate(
            model, prepared["X_test"], prepared["y_test"])
        main_confusion = confusion
        main_metrics["training_time_seconds"] = main_training_seconds
        for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
            print(f"{key.replace('_', ' ').title()}: {main_metrics[key]:.6f}")
        _print_confusion(confusion)

        print("\nSTEP 5: Saving main-model reports and artifacts")
        plot_paths = create_plots(data, features, prepared["y_test"], predicted, scores, confusion)
        save_csv_reports(main_metrics, prepared["y_test"], predicted, scores,
                         data, prepared["train_index"], prepared["test_index"])
        save_excel_reports(dataset_summary, data, features, main_metrics, class_report)
        if config.SAVE_MODEL:
            save_artifacts(model, prepared["imputer"], prepared["scaler"], features)
            print(f"Model and preprocessing artifacts saved in {config.MODEL_DIR}.")
        if config.SAVE_EDUCATIONAL_QP:
            explain_formulation()

        if config.RUN_SVM_EXPERIMENTS and args.experiment != "pca":
            experiments = run_svm_experiments(raw_train_X, y_train, groups, selected=args.experiment)
        elif args.experiment != "pca":
            print("SKIPPED: RUN_SVM_EXPERIMENTS is disabled in config.py")
        if config.RUN_PCA_EXPERIMENT and args.experiment in {"all", "pca"}:
            pca_result = run_pca_experiment(prepared["X_train"], y_train,
                                            prepared["X_test"], prepared["y_test"])
        elif args.experiment in {"all", "pca"}:
            print("SKIPPED: RUN_PCA_EXPERIMENT is disabled in config.py")

        summary_path = config.SUMMARY_DIR / "project_summary.txt"
        summary_path.write_text(
            "CHEMICAL PROCESS FAULT DETECTION USING SVM\n"
            f"Run ID: {run_id}\n"
            f"Dataset: Tennessee Eastman Process ({', '.join(dataset_summary['source_files'])})\n"
            f"Sampled rows: {dataset_summary['sampled_observations']}\n"
            f"Features: {len(features)}\n"
            f"Split: {prepared['split_strategy']}\n"
            f"Training rows: {len(y_train)}\nTesting rows: {len(prepared['y_test'])}\n"
            + "\n".join(f"{key}: {value}" for key, value in main_metrics.items()) + "\n",
            encoding="utf-8")
        copy_main_results_into_project_folder()

    print("\nComputing SHA-256 hashes for available training/testing CSV inputs...")
    input_hashes = collect_input_hashes()
    manifest_path, report_path = write_run_records(
        run_id, dataset_summary, prepared, main_metrics, experiments,
        pca_result, run_kind=("experiments-only" if args.experiments_only else "full"),
        confusion_matrix=main_confusion, input_hashes=input_hashes)
    print(f"\nRun manifest: {manifest_path.relative_to(config.RESULTS_ROOT_DIR)}")
    print(f"Measured run report: {report_path.relative_to(config.RESULTS_ROOT_DIR)}")
    _print_saved_paths(run_dir)
    print("\nPROJECT EXECUTION COMPLETED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
