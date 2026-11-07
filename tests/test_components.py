"""Focused component checks. / Verificações pontuais dos componentes."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import matplotlib as mpl

from bilstm_pipeline.dataset import group_split, validate_records, validate_split
from bilstm_pipeline.execution import load_run
from bilstm_pipeline.metrics import binary_metrics, grouped_bootstrap
from bilstm_pipeline.model import predict
from bilstm_pipeline.plots import plot_binary_metrics, plot_confusion_matrix
from bilstm_pipeline.results import save_evaluation
from bilstm_pipeline.robustness import brightness, contrast, gaussian_noise, occlusion


class ComponentTests(unittest.TestCase):
    """Check reusable behavior. / Verifica comportamentos reutilizáveis."""

    def test_binary_metrics_and_undefined_values(self):
        result = binary_metrics([0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1])
        self.assertEqual(result["confusion_matrix"], [[2, 1], [1, 2]])
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["false_positive_rate"], 1 / 3)
        self.assertAlmostEqual(result["classes"]["1"]["f1"], 2 / 3)
        missing_class = binary_metrics([1, 1], [1, 1])
        self.assertIsNone(missing_class["specificity"])
        self.assertIsNone(missing_class["f1_macro"])

    def test_paired_group_bootstrap(self):
        result = grouped_bootstrap([0, 1, 0, 1], [0, 1, 1, 1], ["a", "a", "b", "b"], paired_pred=[0, 1, 1, 1], n_resamples=20, seed=7)
        self.assertEqual((result["estimate"], result["lower"], result["upper"]), (0.0, 0.0, 0.0))
        self.assertEqual(result["comparison"], "B - A")

    def test_group_split_and_known_duplicates(self):
        records = [{"sample_id": f"s{i}", "video_id": f"v{i // 2}", "group_id": f"g{i // 2}", "source": "synthetic", "label": i % 2} for i in range(12)]
        assignment = group_split(records, seed=7)
        self.assertEqual(assignment, group_split(records, seed=7))
        summary = validate_split(records, assignment)
        self.assertTrue(all(summary[name]["groups"] >= 1 for name in summary))
        wrong = assignment.copy()
        wrong["s1"] = next(name for name in ("train", "validation", "test") if name != wrong["s0"])
        with self.assertRaises(ValueError):
            validate_split(records, wrong)
        records[0]["sha256"] = "a" * 64
        records[2]["sha256"] = "A" * 64
        with self.assertRaises(ValueError):
            validate_records(records)

    def test_perturbations_preserve_shape_and_input(self):
        frames = np.full((3, 6, 6, 3), 128, dtype=np.uint8)
        original = frames.copy()
        masked = occlusion(frames, top=1, left=2, height=2, width=3)
        self.assertTrue(np.all(masked[:, 1:3, 2:5] == 0))
        self.assertTrue(np.array_equal(frames, original))
        self.assertEqual(brightness(frames, 0.1).dtype, np.uint8)
        self.assertEqual(contrast(frames, 1.1).shape, frames.shape)
        self.assertTrue(np.array_equal(gaussian_noise(frames, 0.1, seed=3), gaussian_noise(frames, 0.1, seed=3)))

    def test_adapter_and_artifact_validation(self):
        class Adapter:
            """Example adapter. / Adaptador de exemplo."""
            def predict_scores(self, batch):
                return np.array([0.2, 0.8])
        scores, labels = predict(Adapter(), np.zeros((2, 1)), threshold=0.5)
        self.assertEqual(labels.tolist(), [0, 1])
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "evaluation"
            save_evaluation(path, sample_ids=["a", "b"], y_true=[0, 1], y_pred=labels, scores=scores, threshold=0.5, source="synthetic", unit="sample")
            self.assertEqual(json.loads((path / "evaluation.json").read_text(encoding="utf-8"))["metrics"]["accuracy"], 1)
            with self.assertRaises(FileExistsError):
                save_evaluation(path, sample_ids=["a", "b"], y_true=[0, 1], y_pred=labels, source="synthetic", unit="sample")

    def test_legacy_run_can_be_read(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "output" / "old-run"
            target.mkdir(parents=True)
            (target / "manifesto.json").write_text('{"estado":"concluida"}', encoding="utf-8")
            (target / "config_efetiva.toml").write_text('[experimento]\ntipo="demonstracao_sintetica"\nseed=42\npontos=12\n', encoding="utf-8")
            with patch("bilstm_pipeline.execution.project_root", return_value=Path(root)):
                self.assertTrue(load_run("old-run")["legacy"])

    def test_plot_exports_and_style_isolation(self):
        metrics = binary_metrics([0, 1, 0, 1], [0, 1, 1, 1])
        font_size = mpl.rcParams["font.size"]
        with tempfile.TemporaryDirectory() as root:
            plot_confusion_matrix(metrics, root)
            plot_binary_metrics(metrics, root)
            self.assertTrue(all((Path(root) / f"{name}.{extension}").is_file() for name in ("confusion_matrix", "binary_metrics") for extension in ("pdf", "png")))
        self.assertEqual(mpl.rcParams["font.size"], font_size)


if __name__ == "__main__":
    unittest.main()
