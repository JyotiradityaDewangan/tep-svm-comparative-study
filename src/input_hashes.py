"""Create portable SHA-256 provenance records for CSV inputs."""
from pathlib import Path
import hashlib

import config

OFFICIAL_TEST_FILES = ("TEP_FaultFree_Testing.csv", "TEP_Faulty_Testing.csv")


def _file_record(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def collect_input_hashes(training_dir=None, testing_dir=None):
    """Hash training CSV inputs and expected official test CSVs, if present."""
    training_dir = Path(training_dir or config.DATASET_PATH)
    testing_dir = Path(testing_dir or config.ROOT_DIR / "data" / "testing")
    training_files = sorted(training_dir.rglob("*.csv")) if training_dir.exists() else []
    testing_files = [testing_dir / filename for filename in OFFICIAL_TEST_FILES]
    return {
        "training_csvs": {path.name: _file_record(path) for path in training_files},
        "official_testing_csvs": {
            path.name: _file_record(path) for path in testing_files if path.is_file()
        },
        "missing_official_testing_csvs": [path.name for path in testing_files if not path.is_file()],
    }


def collect_training_hashes(training_dir=None):
    """Hash every training CSV without reading the official test files."""
    training_dir = Path(training_dir or config.DATASET_PATH)
    paths = sorted(training_dir.rglob("*.csv")) if training_dir.exists() else []
    return {path.name: _file_record(path) for path in paths}
