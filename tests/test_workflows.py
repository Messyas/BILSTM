"""Check command workflows. / Verifica fluxos de linha de comando."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from bilstm_pipeline.configuration import project_root


class WorkflowTests(unittest.TestCase):
    """Exercise scripts with synthetic inputs. / Exercita scripts com entradas sintéticas."""

    def test_evaluate_and_split_scripts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.csv"
            with predictions.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["sample_id", "true_label", "predicted_label", "score"])
                writer.writerows([["a", 0, 0, 0.2], ["b", 1, 1, 0.8]])
            evaluation = root / "evaluation"
            command = [sys.executable, str(project_root() / "scripts" / "evaluate.py"), "--input", str(predictions), "--output", str(evaluation), "--source", "synthetic", "--unit", "sample", "--threshold", "0.5"]
            subprocess.run(command, check=True, capture_output=True, text=True)
            result = json.loads((evaluation / "evaluation.json").read_text(encoding="utf-8"))
            self.assertEqual(result["metrics"]["accuracy"], 1.0)
            self.assertTrue((evaluation / "figures" / "confusion_matrix.pdf").is_file())
            duplicate = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(duplicate.returncode, 0)

            records = [{"sample_id": f"s{i}", "video_id": f"v{i}", "group_id": f"g{i}", "source": "synthetic", "label": i % 2} for i in range(6)]
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"dataset_name": "demo", "source": "synthetic", "license": "not applicable", "records": records}), encoding="utf-8")
            result_path = root / "split.json"
            command = [sys.executable, str(project_root() / "scripts" / "split_dataset.py"), "--input", str(manifest), "--output", str(result_path), "--seed", "42"]
            subprocess.run(command, check=True, capture_output=True, text=True)
            split = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(len(split["assignments"]), 6)
            self.assertEqual(set(split["assignments"].values()), {"train", "validation", "test"})
            self.assertEqual(split["split_metadata"]["seed"], 42)


if __name__ == "__main__":
    unittest.main()
