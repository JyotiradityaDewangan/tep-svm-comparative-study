"""Load TEP CSV data safely, identify labels/metadata, and summarize the data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import config

METADATA_HINTS = {"faultnumber", "fault_number", "label", "class", "target", "simulationrun", "sample", "timestamp", "time", "run", "run_id"}

def _target_column(columns):
    normalized = {str(c).strip().lower(): c for c in columns}
    for name in ("faultnumber", "fault_number", "label", "class", "target", "fault", "status"):
        if name in normalized:
            return normalized[name]
    return None

def _read_csv_sample(path, remaining, rng):
    """Read chunks and maintain a uniform, bounded reservoir sample."""
    chunksize = 100_000
    kept = pd.DataFrame()
    seen = 0
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunk["_source_file"] = path.name
        if remaining is None:
            kept = pd.concat([kept, chunk], ignore_index=True)
            seen += len(chunk)
            continue
        # Fill the reservoir, then for each later row choose uniformly from
        # every row seen so far. Replacement slots are applied in input order.
        fill_count = min(max(remaining - len(kept), 0), len(chunk))
        if fill_count:
            kept = pd.concat([kept, chunk.iloc[:fill_count]], ignore_index=True)
        later_rows = chunk.iloc[fill_count:]
        seen += len(chunk)
        if remaining == 0 or later_rows.empty:
            continue
        first_later_row_number = seen - len(later_rows) + 1
        row_numbers = np.arange(first_later_row_number, seen + 1)
        replacement_slots = rng.integers(0, row_numbers, endpoint=False)
        # Keep the last incoming row for each slot when one chunk selects the
        # same reservoir position multiple times.
        last_row_for_slot = {}
        for row_index in np.flatnonzero(replacement_slots < remaining):
            last_row_for_slot[int(replacement_slots[row_index])] = int(row_index)
        if last_row_for_slot:
            slots = np.fromiter(last_row_for_slot.keys(), dtype=int)
            row_indices = np.fromiter(last_row_for_slot.values(), dtype=int)
            # Assign by column in batches to preserve each column's dtype.
            replacement_rows = later_rows.iloc[row_indices]
            for column_index, column_name in enumerate(kept.columns):
                kept.iloc[slots, column_index] = replacement_rows[column_name].to_numpy()
    return kept, seen

def _count_csv_data_rows(path):
    """Count lines in fixed-size blocks before an unsafe full read is attempted."""
    line_count = 0
    with open(path, "rb") as source:
        while True:
            block = source.read(8 * 1024 * 1024)
            if not block:
                break
            line_count += block.count(b"\n")
    return max(0, line_count - 1)  # subtract the header

def load_dataset():
    """Load supported CSV files, infer binary labels, and exclude metadata."""
    directory = Path(config.DATASET_PATH)
    paths = sorted(directory.rglob("*.csv")) if directory.exists() else []
    if not paths:
        raise FileNotFoundError(
            f"No CSV dataset found under {directory}. Download the TEP CSV files and put them in data/raw/. "
            "Required files can be the normal and faulty training CSVs described in README.md."
        )
    if config.SAMPLE_SIZE is None:
        estimated_rows = sum(_count_csv_data_rows(path) for path in paths)
        if estimated_rows > config.MAX_SVC_ROWS:
            raise ValueError(
                f"SAMPLE_SIZE=None would load {estimated_rows:,} rows, above the SVC safety cap "
                f"of {config.MAX_SVC_ROWS:,}. Choose a bounded SAMPLE_SIZE instead."
            )
    rng = __import__("numpy").random.default_rng(config.RANDOM_STATE)
    if config.SAMPLE_SIZE is None:
        budgets = [None] * len(paths)
    else:
        # Divide the requested total exactly across files so the loader never
        # exceeds SAMPLE_SIZE because of rounding.
        quotient, remainder = divmod(config.SAMPLE_SIZE, len(paths))
        budgets = [quotient + (1 if index < remainder else 0) for index in range(len(paths))]
    parts, total_scanned = [], 0
    rows_by_source_file = {}
    for path, per_file_budget in zip(paths, budgets):
        suffix = path.suffix.lower()
        if suffix != ".csv":
            continue
        sample, seen = _read_csv_sample(path, per_file_budget, rng)
        total_scanned += seen
        rows_by_source_file[path.name] = int(seen)
        if sample.empty:
            continue
        parts.append(sample)
    if not parts:
        raise ValueError("CSV files were found, but none contained data rows.")
    data = pd.concat(parts, ignore_index=True)
    raw_columns = [column for column in data.columns if not str(column).startswith("_")]
    # Preserve source path as metadata so the split can keep simulation runs together.
    target = _target_column(data.columns)
    if target is None:
        # Separate fault-free/faulty named files can provide the label.
        source = data["_source_file"].str.lower()
        if source.str.contains("faultfree|normal").any() and source.str.contains("faulty|fault_").any():
            data["_inferred_fault_label"] = source.str.contains("faulty|fault_").astype(int)
            target = "_inferred_fault_label"
        else:
            raise ValueError("Could not identify a target column. Expected faultNumber/label/class or files named normal/faulty.")
    raw_labels = pd.to_numeric(data[target], errors="coerce")
    if raw_labels.isna().any():
        raise ValueError(f"Target column '{target}' contains missing or non-numeric labels.")
    labels = sorted(raw_labels.unique().tolist())
    if config.NORMAL_LABEL not in labels:
        raise ValueError(f"NORMAL_LABEL={config.NORMAL_LABEL!r} is absent from '{target}'. Observed labels: {labels[:30]}")
    if len(labels) < 2:
        raise ValueError(f"Only one label ({labels[0]!r}) was loaded; include both normal and faulty observations.")
    # This project deliberately collapses all verified non-normal fault IDs into class 1.
    data["_binary_target"] = (raw_labels != config.NORMAL_LABEL).astype("int8")
    metadata = [c for c in data.columns if str(c).strip().lower() in METADATA_HINTS or str(c).startswith("_") or c == target]
    source_metadata = [column for column in metadata if column in raw_columns]
    internal_columns = [column for column in metadata if column not in raw_columns]
    features = [c for c in data.columns if c not in metadata and pd.api.types.is_numeric_dtype(data[c])]
    if not features:
        raise ValueError("No numeric process measurement columns were found after metadata removal.")
    if not config.USE_ALL_FEATURES:
        missing = [name for name in config.SELECTED_FEATURES if name not in features]
        if missing:
            raise ValueError(f"Configured feature columns are missing: {missing}")
        features = config.SELECTED_FEATURES.copy()
    summary = {
        "dataset_name": "Tennessee Eastman Process (provided CSV files)",
        "sampled_observations": int(len(data)), "rows_scanned": int(total_scanned),
        "rows_by_source_file": rows_by_source_file,
        "number_of_columns": int(len(raw_columns)), "target_column": str(target),
        "input_features": features, "number_of_input_features": len(features),
        "metadata_columns": [str(c) for c in source_metadata],
        "internal_columns_excluded": [str(c) for c in internal_columns],
        "normal_label_in_source": config.NORMAL_LABEL,
        "normal_observations": int((data["_binary_target"] == 0).sum()),
        "faulty_observations": int((data["_binary_target"] == 1).sum()),
        "missing_values": {str(k): int(v) for k,v in data.isna().sum().items() if v},
        "duplicate_rows": int(data[raw_columns].duplicated().sum()),
        "quality_check_scope": "missing values and duplicates checked on the sampled rows",
        "source_files": [p.name for p in paths],
        "sampled_class_distribution_by_source": {
            str(source): {
                "NORMAL": int(np.sum((data["_source_file"] == source) & (data["_binary_target"] == 0))),
                "FAULTY": int(np.sum((data["_source_file"] == source) & (data["_binary_target"] == 1))),
            }
            for source in sorted(data["_source_file"].unique())
        },
    }
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.PROCESSED_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Loaded sample: {len(data):,} rows, {len(raw_columns)} source columns, "
          f"{len(features)} model features; scanned {total_scanned:,} rows.")
    print("Source CSV columns:", ", ".join(map(str, raw_columns)))
    print("First five source rows:\n", data[raw_columns].head().to_string(index=False))
    print(f"Target: {target}; source metadata excluded: {source_metadata}; "
          f"internal metadata excluded: {internal_columns}")
    print(f"Normal={summary['normal_observations']:,}, faulty={summary['faulty_observations']:,}, duplicates={summary['duplicate_rows']:,}")
    return data, features, summary
