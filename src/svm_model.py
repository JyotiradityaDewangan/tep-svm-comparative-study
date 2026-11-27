"""Build and persist the project's single SVM classifier."""
from pathlib import Path
import joblib
from sklearn.svm import SVC
import config

def make_model(kernel=None):
    """SVM separates classes with a maximum-margin boundary; kernels allow curved boundaries."""
    return SVC(kernel=kernel or config.SVM_KERNEL, C=config.SVM_C, gamma=config.SVM_GAMMA,
               class_weight="balanced" if config.USE_CLASS_BALANCING else None,
               decision_function_shape="ovr", random_state=config.RANDOM_STATE)

def save_artifacts(model, imputer, scaler, feature_names):
    """Save everything needed for matching future predictions."""
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_DIR / "trained_svm_model.pkl")
    joblib.dump(imputer, config.MODEL_DIR / "imputer.pkl")
    joblib.dump(scaler, config.MODEL_DIR / "scaler.pkl")
    joblib.dump(list(feature_names), config.MODEL_DIR / "feature_names.pkl")

def load_artifacts(model_dir=None):
    """Load the classifier, imputer, scaler, and exact feature order."""
    model_dir = Path(model_dir) if model_dir is not None else config.MODEL_DIR
    model_path = model_dir / "trained_svm_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}. Run python main.py first.")
    model = joblib.load(model_path)
    imputer = joblib.load(model_dir / "imputer.pkl")
    scaler_path = model_dir / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    feature_names = joblib.load(model_dir / "feature_names.pkl")
    return model, imputer, scaler, feature_names
