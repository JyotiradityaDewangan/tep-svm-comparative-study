"""Split observations first, then fit imputation and scaling on training data."""
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import config

def fit_preprocessor_on_training(X_fit_raw, X_validation_raw):
    """Fit imputation/scaling on fit rows only, then transform validation rows.

    This helper is used by secondary experiments, whose internal validation
    split is made inside the main training partition.
    """
    X_fit_raw = pd.DataFrame(X_fit_raw).apply(pd.to_numeric, errors="coerce")
    if X_validation_raw is not None:
        X_validation_raw = pd.DataFrame(X_validation_raw).apply(pd.to_numeric, errors="coerce")
        if list(X_fit_raw.columns) != list(X_validation_raw.columns):
            raise ValueError("Fitting and validation features must have the same columns and order.")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_fit = imputer.fit_transform(X_fit_raw)
    X_validation = None if X_validation_raw is None else imputer.transform(X_validation_raw)
    scaler = None
    if config.NORMALIZE_DATA:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X_fit)
        if X_validation is not None:
            X_validation = scaler.transform(X_validation)
    return X_fit, X_validation, imputer, scaler

def prepare_data(data, feature_names):
    """Return arrays and fitted transformers; simulation runs stay in one split where feasible."""
    X = data[feature_names].apply(pd.to_numeric, errors="coerce")
    y = data["_binary_target"].astype(int)
    groups = None
    run_col = next((c for c in data.columns if str(c).lower() in {"simulationrun", "simulation_run", "run_id"}), None)
    if run_col:
        source = data.get("_source_file", pd.Series("dataset", index=data.index)).astype(str)
        groups = source + "::" + data[run_col].astype(str)
    split_used = "stratified random split"
    train_idx = test_idx = None
    if groups is not None and groups.nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=30, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
        for candidate_train, candidate_test in splitter.split(X, y, groups):
            if y.iloc[candidate_train].nunique() == 2 and y.iloc[candidate_test].nunique() == 2:
                train_idx, test_idx = candidate_train, candidate_test
                break
        if train_idx is not None:
            split_used = "group-aware split by source file and simulation run"
    if train_idx is None:
        train_idx, test_idx = train_test_split(
            range(len(X)), test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
            stratify=y if y.value_counts().min() >= 2 else None,
        )
        if groups is not None:
            split_used += " (group split could not keep both classes in each partition)"
    X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    # Median values are learned only from training rows, preventing test leakage.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train = imputer.fit_transform(X_train_raw)
    X_test = imputer.transform(X_test_raw)
    scaler = None
    if config.NORMALIZE_DATA:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Both NORMAL and FAULTY classes are required in training and test data.")
    train_groups = groups.iloc[train_idx].to_numpy() if groups is not None else None
    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train.to_numpy(), "y_test": y_test.to_numpy(),
        "train_index": list(train_idx), "test_index": list(test_idx), "imputer": imputer, "scaler": scaler,
        "split_strategy": split_used, "feature_names": feature_names,
        "train_groups": train_groups,
    }
