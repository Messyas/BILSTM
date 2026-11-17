"""Protocol checks: threshold, label conflicts, latency. / Verificações do protocolo."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bilstm_pipeline.research_calibration import select_threshold
from bilstm_pipeline.research_data import _drop_label_conflicts


class ThresholdSelectionTests(unittest.TestCase):
    def test_fpr_cap_and_recall(self):
        labels = [0] * 20 + [1] * 10
        negatives = list(np.linspace(0.0, 0.95, 20))
        positives = [0.9, 0.92, 0.97, 0.98, 0.99, 0.5, 0.6, 0.7, 0.8, 0.85]
        result = select_threshold(labels, negatives + positives, 0.05)
        self.assertEqual(result["allowed_false_positives"], 1)
        self.assertLessEqual(result["validation_fpr"], 0.05)
        self.assertEqual(result["validation_false_positives"], 1)
        predictions = (np.asarray(negatives + positives) >= result["threshold"]).astype(int)
        self.assertEqual(int(predictions[20:].sum()), result["validation_true_positives"])
        # Any lower threshold would admit a second false positive.
        second = sorted(negatives)[-2]
        self.assertGreater(result["threshold"], second)

    def test_zero_target_and_ties(self):
        result = select_threshold([0, 0, 1, 1], [0.7, 0.7, 0.7, 0.9], 0.0)
        self.assertEqual(result["validation_false_positives"], 0)
        self.assertEqual(result["validation_true_positives"], 1)
        self.assertAlmostEqual(result["threshold"], 0.8)

    def test_unattainable_when_saturated(self):
        result = select_threshold([0, 1], [1.0, 1.0], 0.0)
        self.assertFalse(result["target_attained"])
        self.assertLessEqual(result["threshold"], 1.0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            select_threshold([1, 1], [0.2, 0.3], 0.05)
        with self.assertRaises(ValueError):
            select_threshold([0, 1], [0.2, 1.3], 0.05)


class LabelConflictTests(unittest.TestCase):
    def test_conflicting_group_removed(self):
        records = [
            {"sample_id": "NonViolence/a.mp4", "label": 0, "group_id": "g1", "sha256": "1"},
            {"sample_id": "Violence/b.mp4", "label": 1, "group_id": "g1", "sha256": "1"},
            {"sample_id": "Violence/c.mp4", "label": 1, "group_id": "g2", "sha256": "2"},
            {"sample_id": "Violence/d.mp4", "label": 1, "group_id": "g2", "sha256": "2"},
        ]
        kept, excluded = _drop_label_conflicts(records)
        self.assertEqual([record["sample_id"] for record in kept], ["Violence/c.mp4", "Violence/d.mp4"])
        self.assertEqual({item["sample_id"] for item in excluded}, {"NonViolence/a.mp4", "Violence/b.mp4"})


class StreamingLatencyTests(unittest.TestCase):
    def test_stream_on_synthetic_video(self):
        import cv2
        import torch
        from bilstm_pipeline.research_latency import _stream

        class Constant(torch.nn.Module):
            def forward(self, videos):
                return torch.zeros(videos.shape[0], 2)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "Violence" / "v.avi"
            path.parent.mkdir()
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (32, 32))
            for index in range(90):
                writer.write(np.full((32, 32, 3), index % 255, dtype=np.uint8))
            writer.release()
            record = {"sample_id": "Violence/v.avi", "path": "Violence/v.avi"}
            config = {"video": {"frames": 15, "width": 10, "height": 10}}
            result = _stream([record], Path(temporary), config, torch.device("cpu"), Constant(), fps=30, window_seconds=2, stride_seconds=0.5)
        # 90 frames, window 60, stride 15 -> decisions at frames 60, 75, 90.
        self.assertEqual(result["per_decision_total"]["n"], 3)
        self.assertEqual(result["per_frame_decode"]["n"], 90)
        self.assertEqual(result["protocol"]["frame_offsets"][-1], 56)


if __name__ == "__main__":
    unittest.main()
