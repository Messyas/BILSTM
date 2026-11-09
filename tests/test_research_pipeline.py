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

            rwf_raw = root / "rwf2000-v1"
            for split_dir in ("train", "val"):
                for label, folder in enumerate(("NonFight", "Fight")):
                    directory = rwf_raw / split_dir / folder
                    directory.mkdir(parents=True)
                    for index in range(3):
                        (directory / f"{split_dir}-{index}.avi").write_bytes(f"rwf-{split_dir}-{label}-{index}".encode())
            ext_config = load_research_config("configs/rwf2000_external.toml")
            ext_config["data"] = {"root": str(rwf_raw), "manifest": str(root / "rwf2000.json")}
            ext_config_path = root / "rwf_config.toml"
            with ext_config_path.open("wb") as stream:
                tomli_w.dump(ext_config, stream)
            rwf_manifest = load_manifest(prepare_dataset(ext_config_path))
            self.assertEqual(rwf_manifest["dataset_name"], "RWF-2000")
            self.assertEqual(rwf_manifest["source"], "vulamnguyen/rwf2000")
            self.assertEqual(len(rwf_manifest["records"]), 12)

    def test_reviewed_groups_are_kept_together_and_mapping_is_complete(self):
        from bilstm_pipeline.research_data import _apply_groups, inventory_videos
        from bilstm_pipeline.dataset import validate_split
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label, folder in enumerate(("NonViolence", "Violence")):
                directory = root / folder
                directory.mkdir()
                for index in range(6):
                    (directory / f"v{index}.mp4").write_bytes(f"{label}-{index}".encode())
            records = inventory_videos(root)
            mapping = root / "groups.csv"
            rows = ["sample_id,group_id"]
            for record in records:
                folder = Path(record["path"]).parts[0]
                number = int(Path(record["path"]).stem[1:])
                rows.append(f"{record['sample_id']},{folder}-{number // 2}")
            mapping.write_text("\n".join(rows), encoding="utf-8")
            config = load_research_config("configs/rlvs_baseline.toml")
            config["data"] = {"root": str(root), "manifest": str(root / "manifest.json")}
            config_path = root / "config.toml"
            with config_path.open("wb") as stream:
                tomli_w.dump(config, stream)
            manifest = load_manifest(prepare_dataset(config_path, groups_path=mapping))
            self.assertIsNotNone(manifest["split_metadata"]["group_mapping_sha256"])
            validate_split(manifest["records"], manifest["assignments"])
            for group in {record["group_id"] for record in manifest["records"]}:
                self.assertEqual(len({manifest["assignments"][r["sample_id"]] for r in manifest["records"] if r["group_id"] == group}), 1)
            mapping.write_text("\n".join(rows[:-1]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly all"):
                _apply_groups(records, mapping)

    def test_reviewed_mapping_rejects_identical_content_in_distinct_groups(self):
        from bilstm_pipeline.research_data import _apply_groups
        digest = "a" * 64
        records = [{"sample_id": name, "video_id": name, "group_id": name, "source": "test", "label": 0, "sha256": digest} for name in ("a", "b")]
        with tempfile.TemporaryDirectory() as temporary:
            mapping = Path(temporary) / "groups.csv"
            mapping.write_text("sample_id,group_id\na,one\nb,two\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate content"):
                _apply_groups(records, mapping)


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
            writer = cv2.VideoWriter(str(path), getattr(cv2, "VideoWriter_fourcc")(*"MJPG"), 5, (32, 32))
            self.assertTrue(writer.isOpened())
            for index in range(4):
                writer.write(np.full((32, 32, 3), index * 40, dtype=np.uint8))
            writer.release()
            frames = decode_video(path, frames=6, width=32, height=32)
            self.assertEqual(frames.shape, (6, 32, 32, 3))
            self.assertEqual([round(float(frame.mean()) * 255) for frame in frames], [0, 0, 40, 40, 80, 120])
            self.assertEqual(tuple(video_tensor(frames).shape), (6, 3, 32, 32))
            self.assertEqual(jpeg_compression(frames, quality=50).shape, frames.shape)
        for backbone in (
            "mobilenet_v2",
            "mobilenet_v3_large",
            "densenet121",
            "inception_v3",
            "efficientnet_v2_s",
            "convnext_tiny",
            "swin_t",
            "vit_b_16",
        ):
            for attention in (True, False):
                model = ConflictModel(backbone=backbone, hidden_size=4, dense_units=(8, 4), dropout=0.0, pretrained=False, attention=attention).eval()
                with torch.inference_mode():
                    logits = model(torch.zeros((1, 2, 3, 32, 32)))
                    model(torch.ones((1, 2, 3, 32, 32)))
                    repeated = model(torch.zeros((1, 2, 3, 32, 32)))
                self.assertEqual(tuple(logits.shape), (1, 2))
                self.assertTrue(torch.equal(logits, repeated))
        for attention in (True, False):
            tf_model = ConflictModel(backbone="swin_t", temporal_head="transformer", hidden_size=4, dense_units=(8, 4), dropout=0.0, pretrained=False, attention=attention).eval()
            with torch.inference_mode():
                tf_logits = tf_model(torch.zeros((1, 2, 3, 32, 32)))
            self.assertEqual(tuple(tf_logits.shape), (1, 2))
        densenet_flat = ConflictModel(backbone="densenet121", pooling="flatten", hidden_size=32, dense_units=(256, 128, 64, 32), pretrained=False, frames=15, width=100, height=100)
        self.assertEqual(densenet_flat.recurrent.input_size, 9216)
        self.assertEqual(sum(p.numel() for p in densenet_flat.recurrent.parameters()), 2368000)
        self.assertIsNotNone(densenet_flat.attention)
        assert densenet_flat.attention is not None
        self.assertEqual(sum(p.numel() for p in densenet_flat.attention.parameters()), 79)
        self.assertEqual(sum(p.numel() for p in densenet_flat.classifier.parameters()), 59938)

    def test_decode_releases_capture_when_second_pass_fails(self):
        from bilstm_pipeline.research_data import decode_video
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        first = Mock()
        first.isOpened.return_value = True
        first.read.side_effect = [(True, image), (True, image), (False, None)]
        second = Mock()
        second.isOpened.return_value = True
        second.read.side_effect = [(True, image), (False, None)]
        with patch("cv2.VideoCapture", side_effect=[first, second]):
            with self.assertRaisesRegex(ValueError, "changed between passes"):
                decode_video(Path("video.mp4"), frames=2, width=8, height=8)
        first.release.assert_called_once()
        second.release.assert_called_once()

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
            original = ConflictModel(**config["model"], **config["video"])
            torch.save({"model_state": original.state_dict()}, root / "best.pt")
            with patch("bilstm_pipeline.research_evaluation.run_path", return_value=root):
                _, _, device, loaded = _load_run("example", None)
            self.assertEqual(device.type, "cpu")
            self.assertTrue(torch.equal(original.state_dict()["classifier.0.weight"], loaded.state_dict()["classifier.0.weight"]))

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
            with patch("bilstm_pipeline.research_training.run_path", side_effect=lambda run_id: run_root / run_id), patch("bilstm_pipeline.research_training.ConflictModel", return_value=torch.nn.Linear(1, 2)), patch("bilstm_pipeline.research_training._run_epoch", side_effect=[(0.8, 0.6), (0.5, 0.9)]), patch("bilstm_pipeline.research_training._device_metadata", return_value={"device": "mock"}):
                run_id = train_model(config_path, backbone="densenet121", attention=False, batch_size=64, min_learning_rate=0.00005)
            target = run_root / run_id
            status = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "completed")
            self.assertAlmostEqual(status["best_validation_accuracy"], 0.9)
            self.assertTrue((target / "best.pt").is_file())
            saved_config = load_research_config(target / "effective_config.toml")
            self.assertEqual(saved_config["model"]["backbone"], "densenet121")
            self.assertFalse(saved_config["model"]["attention"])
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
