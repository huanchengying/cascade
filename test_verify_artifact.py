import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from verify_artifact_20261008 import audit_results


class ArtifactCoverageTests(unittest.TestCase):
    def test_duplicate_cannot_replace_missing_matrix_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "algorithm.py"
            source.write_text("pass\n")
            payload = {
                "metadata": {
                    "config": {
                        "datasets": [{"name": "d", "roots": ["r"]}],
                        "methods": ["a", "b"],
                        "budget_s": 10,
                        "repeats": 1,
                    },
                    "sha256": {"algorithm.py": hashlib.sha256(source.read_bytes()).hexdigest()},
                },
                "records": [
                    {"dataset": "d", "root": "r", "method": "a", "budget_s": 10, "repeat": 0},
                    {"dataset": "d", "root": "r", "method": "a", "budget_s": 10, "repeat": 0},
                ],
            }
            results = base / "results.json"
            results.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                audit_results(results, base)


if __name__ == "__main__":
    unittest.main()
