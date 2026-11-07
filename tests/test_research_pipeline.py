"""Check research infrastructure without training. / Verifica infraestrutura sem treinar."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import tomli_w

from bilstm_pipeline.research_config import load_research_config
from bilstm_pipeline.research_data import load_manifest, prepare_dataset

_HAS_TRAINING = all(importlib.util.find_spec(name) for name in ("torch", "torchvision", "cv2"))


class PreparationTests(unittest.TestCase):
    """Validate manifest construction. / Valida construção do manifesto."""

    def test_grouped_stratified_manifest_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            for label, folder in enumerate(("NonViolence", "Violence")):
                directory = raw / folder
                directory.mkdir(parents=True)
                for index in range(5):
                    (directory / f"video-{index}.mp4").write_bytes(f"video-{label}-{index}".encode())
            config = load_research_config("configs/rlvs_baseline.toml")
            config["data"] = {"root": str(raw), "manifest": str(root / "manifest.json")}
            config_path = root / "config.toml"
            with config_path.open("wb") as stream:
                tomli_w.dump(config, stream)
            output = prepare_dataset(config_path)
            manifest = load_manifest(output)
            self.assertEqual(len(manifest["records"]), 10)
            for summary in manifest["split_summary"].values():
                self.assertGreater(summary["labels"]["0"], 0)
                self.assertGreater(summary["labels"]["1"], 0)
            with self.assertRaises(FileExistsError):
                prepare_dataset(config_path)


@unittest.skipUnless(_HAS_TRAINING, "Install the training dependency group to check the research pipeline.")
class ResearchPipelineTests(unittest.TestCase):
    """Use inference and mocks only. / Usa apenas inferência e mocks."""

    def test_decode_video_and_model_forward(self):
        import cv2
        import torch
        from bilstm_pipeline.research_data import decode_video
        from bilstm_pipeline.research_model import ConflictModel
        from bilstm_pipeline.research_training import video_tensor
        from bilstm_pipeline.robustness import jpeg_compression
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "example.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (32, 32))
            self.assertTrue(writer.isOpened())
            for index in range(4):
                writer.write(np.full((32, 32, 3), index * 40, dtype=np.uint8))
            writer.release()
            frames = decode_video(path, frames=6, width=32, height=32)
            self.assertEqual(frames.shape, (6, 32, 32, 3))
            self.assertEqual(tuple(video_tensor(frames).shape), (6, 3, 32, 32))
            self.assertEqual(jpeg_compression(frames, quality=50).shape, frames.shape)
        model = ConflictModel(hidden_size=4, dropout=0.0, pretrained=False).eval()
        with torch.inference_mode():
            logits = model(torch.zeros((1, 2, 3, 32, 32)))
        self.assertEqual(tuple(logits.shape), (1, 2))

    def test_compression_command_with_mocked_ffmpeg(self):
        from bilstm_pipeline.research_evaluation import _compressed_video_frames
        frames = np.zeros((2, 16, 16, 3), dtype=np.float32)
        with patch("bilstm_pipeline.research_evaluation.subprocess.run") as process, patch("bilstm_pipeline.research_evaluation.decode_video", return_value=frames):
            result = _compressed_video_frames(Path("source.mp4"), ffmpeg="ffmpeg", crf=28, video_config={"frames": 2, "width": 16, "height": 16})
        self.assertEqual(result.shape, frames.shape)
        command = process.call_args.args[0]
        self.assertEqual(command[command.index("-c:v") + 1], "libx264")
        self.assertEqual(command[command.index("-crf") + 1], "28")

    def test_untrained_checkpoint_roundtrip(self):
        import torch
        from bilstm_pipeline.research_evaluation import _load_run
        from bilstm_pipeline.research_model import ConflictModel
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_research_config("configs/rlvs_baseline.toml")
            config["model"].update(hidden_size=4, pretrained=False)
            config["training"]["device"] = "cpu"
            with (root / "effective_config.toml").open("wb") as stream:
                tomli_w.dump(config, stream)
            (root / "manifest.json").write_text('{"status":"completed","type":"proposed_reconstruction"}', encoding="utf-8")
            original = ConflictModel(**config["model"])
            torch.save({"model_state": original.state_dict()}, root / "best.pt")
            with patch("bilstm_pipeline.research_evaluation.run_path", return_value=root):
                _, _, device, loaded = _load_run("example", None)
            self.assertEqual(device.type, "cpu")
            self.assertTrue(torch.equal(original.classifier[1].weight, loaded.classifier[1].weight))

    def test_orchestration_with_mocked_training_and_inference(self):
        import torch
        from bilstm_pipeline.research_evaluation import evaluate_model
        from bilstm_pipeline.research_training import train_model
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            for label, folder in enumerate(("NonViolence", "Violence")):
                directory = raw / folder
                directory.mkdir(parents=True)
                for index in range(5):
                    (directory / f"v{index}.mp4").write_bytes(f"sample-{label}-{index}".encode())
            config = load_research_config("configs/rlvs_baseline.toml")
            config["data"] = {"root": str(raw), "manifest": str(root / "dataset.json")}
            config["training"].update(device="cpu", epochs=1)
            config["model"]["pretrained"] = False
            config["evaluation"]["bootstrap_resamples"] = 10
            config_path = root / "config.toml"
            with config_path.open("wb") as stream:
                tomli_w.dump(config, stream)
            prepare_dataset(config_path)
            run_root = root / "runs"
            with patch("bilstm_pipeline.research_training.run_path", side_effect=lambda run_id: run_root / run_id), patch("bilstm_pipeline.research_training.ConflictModel", return_value=torch.nn.Linear(1, 2)), patch("bilstm_pipeline.research_training._run_epoch", side_effect=[0.8, 0.5]), patch("bilstm_pipeline.research_training._device_metadata", return_value={"device": "mock"}):
                run_id = train_model(config_path)
            target = run_root / run_id
            status = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "completed")
            self.assertTrue((target / "best.pt").is_file())
            saved_config = load_research_config(target / "effective_config.toml")
            fake_frames = np.full((15, 10, 10, 3), 0.5, dtype=np.float32)
            with patch("bilstm_pipeline.research_evaluation._load_run", return_value=(target, saved_config, torch.device("cpu"), object())), patch("bilstm_pipeline.research_evaluation.decode_video", return_value=fake_frames), patch("bilstm_pipeline.research_evaluation._predict", return_value=(0.5, 0.01, 0.02)):
                evaluation = evaluate_model(run_id, robustness=True)
            self.assertEqual(json.loads((evaluation / "manifest.json").read_text(encoding="utf-8"))["status"], "completed")
            self.assertTrue((evaluation / "clean" / "predictions.csv").is_file())
            self.assertTrue((evaluation / "figures" / "robustness.png").is_file())
            ffmpeg_info = Mock(stdout="ffmpeg mock version\n")
            with patch("bilstm_pipeline.research_evaluation._load_run", return_value=(target, saved_config, torch.device("cpu"), object())), patch("bilstm_pipeline.research_evaluation.decode_video", return_value=fake_frames), patch("bilstm_pipeline.research_evaluation._predict", return_value=(0.5, 0.01, 0.02)), patch("bilstm_pipeline.research_evaluation.shutil.which", return_value="ffmpeg"), patch("bilstm_pipeline.research_evaluation.subprocess.run", return_value=ffmpeg_info), patch("bilstm_pipeline.research_evaluation._compressed_video_frames", return_value=fake_frames):
                compressed = evaluate_model(run_id, robustness=True, video_compression=True)
            self.assertTrue((compressed / "h264_video" / "predictions.csv").is_file())
            self.assertEqual(json.loads((compressed / "manifest.json").read_text(encoding="utf-8"))["ffmpeg_version"], "ffmpeg mock version")
            duplicate = load_manifest(target / "dataset_manifest.json")
            duplicate["source"] = "another-source"
            for record in duplicate["records"]:
                record["sha256"] = record["sha256"].upper()
            external = root / "external.json"
            external.write_text(json.dumps(duplicate), encoding="utf-8")
            with patch("bilstm_pipeline.research_evaluation._load_run", return_value=(target, saved_config, torch.device("cpu"), object())):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    evaluate_model(run_id, dataset_root=raw, external_manifest=external)


if __name__ == "__main__":
    unittest.main()
