"""Predict NORMAL or FAULTY from a new sensor CSV using saved training artifacts."""
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import joblib
import config
from src.svm_model import load_artifacts

def predict_file(input_path, output_path=None, model_dir=None):
    # Verify that a training run has saved a model before parsing the input.
    model,imputer,scaler,features=load_artifacts(model_dir)
    try:
        frame=pd.read_csv(input_path)
    except Exception as exc:
        raise ValueError(f"Could not read prediction CSV '{input_path}': {exc}") from exc
    missing=[name for name in features if name not in frame.columns]
    if missing:
        raise ValueError(f"Prediction CSV is missing required sensor columns: {missing}")
    try:
        X=frame[features].apply(pd.to_numeric,errors="raise")
        X=imputer.transform(X)
        if scaler is not None: X=scaler.transform(X)
    except Exception as exc:
        raise ValueError(f"Sensor values could not be preprocessed: {exc}") from exc
    labels=model.predict(X)
    result=pd.DataFrame({"predicted_class":["NORMAL" if v==0 else "FAULTY" for v in labels]})
    result["decision_score"]=model.decision_function(X)
    if output_path is None:
        output_dir = Path(config.PREDICTIONS_DIR)
        output_dir.mkdir(parents=True,exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_path=output_dir/f"new_data_predictions_{stamp}.csv"
    else:
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(output_path,index=False)
    print(f"Predicted {len(result):,} rows; saved to {output_path}")
    return result
