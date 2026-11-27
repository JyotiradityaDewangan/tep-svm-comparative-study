import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import config


class MissingDatasetCLITests(unittest.TestCase):
    def test_normal_entrypoint_reports_missing_dataset_without_tep_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "project"
            root.mkdir()
            shutil.copy2(config.ROOT_DIR / "main.py", root / "main.py")
            shutil.copy2(config.ROOT_DIR / "config.py", root / "config.py")
            shutil.copytree(config.ROOT_DIR / "src", root / "src",
                            ignore=shutil.ignore_patterns("__pycache__"))
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            result = subprocess.run([sys.executable, "main.py"], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 1)
            self.assertIn("No CSV dataset found", result.stderr)
            self.assertIn("data/raw/", result.stderr)


if __name__ == "__main__":
    unittest.main()
