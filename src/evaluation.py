"""Calculate binary classification metrics and save human-readable reports."""
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix)
import config

def evaluate(model, X_test, y_test):
    started = time.perf_counter()
    predicted = model.predict(X_test)
    test_time = time.perf_counter() - started
    scores = model.decision_function(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predicted),
        "precision": precision_score(y_test, predicted, zero_division=0),
        "recall": recall_score(y_test, predicted, zero_division=0),
        "f1_score": f1_score(y_test, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores) if len(np.unique(y_test)) == 2 else np.nan,
        "testing_time_seconds": test_time,
    }
    return metrics, predicted, scores, classification_report(y_test, predicted, labels=[0,1], target_names=["NORMAL", "FAULTY"], output_dict=True, zero_division=0), confusion_matrix(y_test, predicted, labels=[0,1])

def save_csv_reports(metrics, y_test, predicted, scores, data, train_idx, test_idx):
    config.CSV_DIR.mkdir(parents=True, exist_ok=True)
    if config.SAVE_CSV_REPORTS:
        pd.DataFrame([metrics]).to_csv(config.CSV_DIR / "model_metrics.csv", index=False)
        counts = data["_binary_target"].map({0:"NORMAL",1:"FAULTY"}).value_counts().rename_axis("class").reset_index(name="observations")
        counts.to_csv(config.CSV_DIR / "dataset_class_distribution.csv", index=False)
        report = pd.DataFrame({"actual_class": np.where(y_test==0,"NORMAL","FAULTY"), "predicted_class": np.where(predicted==0,"NORMAL","FAULTY"), "decision_score": scores})
        report["prediction_result"] = np.where(y_test == predicted, "correct", "incorrect")
        report.to_csv(config.CSV_DIR / "actual_vs_predicted.csv", index=False)
    if config.SAVE_PREDICTIONS:
        config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        report = pd.DataFrame({"actual_class": np.where(y_test==0,"NORMAL","FAULTY"), "predicted_class": np.where(predicted==0,"NORMAL","FAULTY"), "prediction_result": np.where(y_test == predicted, "correct", "incorrect"), "decision_score": scores})
        report.to_csv(config.PREDICTIONS_DIR / "test_predictions.csv", index=False)

def save_excel_reports(summary, data, features, metrics, class_report):
    if not config.SAVE_EXCEL_REPORTS:
        return
    config.EXCEL_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = [{"item": k, "value": str(v)} for k,v in summary.items() if k not in {"input_features", "metadata_columns", "missing_values", "source_files"}]
    summary_rows += [{"item":"missing_values", "value":str(summary["missing_values"])}]
    stats = data[features].describe().T.reset_index().rename(columns={"index":"sensor", "50%":"median", "std":"standard_deviation", "min":"minimum", "max":"maximum"})
    stats = stats[["sensor","mean","median","minimum","maximum","standard_deviation"]]
    reports = {
        "dataset_summary.xlsx": {"Dataset":pd.DataFrame(summary_rows)},
        "sensor_statistics.xlsx": {"Sensor Statistics":stats},
        "model_performance.xlsx": {"Metrics":pd.DataFrame([metrics])},
        "classification_report.xlsx": {"Class Metrics":pd.DataFrame(class_report).T.reset_index().rename(columns={"index":"class"})},
    }
    for filename, sheets in reports.items():
        with pd.ExcelWriter(config.EXCEL_DIR / filename, engine="openpyxl") as writer:
            for sheet, frame in sheets.items():
                frame.to_excel(writer, sheet_name=sheet, index=False)
                ws = writer.sheets[sheet]
                ws.freeze_panes = "A2"
                for column in ws.columns:
                    width = min(max(max(len(str(cell.value or "")) for cell in column) + 2, 12), 48)
                    ws.column_dimensions[column[0].column_letter].width = width
