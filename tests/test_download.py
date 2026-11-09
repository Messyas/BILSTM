"""Dataset download checks. / Verificações do download do dataset."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from bilstm_pipeline.download import DATASET_REF, RWF2000_DATASET_REF, download_dataset


class DownloadTests(unittest.TestCase):
    """Use small local archives. / Usa arquivos locais pequenos."""

    def test_success_records_source_and_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "rlvs-v1"
            captured = {}

            def fake_download(command, *, check):
                self.assertTrue(check)
                captured["reference"] = command[3]
                archive = Path(command[5]) / "dataset.zip"
                with zipfile.ZipFile(archive, "w") as zipped:
                    zipped.writestr("Violence/example.mp4", b"synthetic bytes")
                captured["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()

            with patch("bilstm_pipeline.download.shutil.which", return_value="kaggle"), patch("bilstm_pipeline.download.subprocess.run", side_effect=fake_download):
                self.assertEqual(download_dataset(output=target), target.resolve())
            metadata = json.loads((target / "download_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(captured["reference"], DATASET_REF + "/1")
            self.assertEqual(metadata["version"], 1)
            self.assertEqual(metadata["archive_sha256"], captured["sha256"])
            self.assertEqual((target / "Violence" / "example.mp4").read_bytes(), b"synthetic bytes")

            rwf_target = Path(temporary) / "rwf2000-v1"
            with patch("bilstm_pipeline.download.shutil.which", return_value="kaggle"), patch("bilstm_pipeline.download.subprocess.run", side_effect=fake_download):
                self.assertEqual(download_dataset(dataset="rwf2000", output=rwf_target), rwf_target.resolve())
            self.assertEqual(captured["reference"], RWF2000_DATASET_REF + "/1")

    def test_existing_destination_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "existing"
            target.mkdir()
            marker = target / "existing.txt"
            marker.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                download_dataset(output=target)
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_unsafe_archive_and_failed_download_leave_no_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "dataset"

            def fake_unsafe(command, *, check):
                with zipfile.ZipFile(Path(command[5]) / "dataset.zip", "w") as zipped:
                    zipped.writestr("../escape.txt", "unsafe")

            with patch("bilstm_pipeline.download.shutil.which", return_value="kaggle"), patch("bilstm_pipeline.download.subprocess.run", side_effect=fake_unsafe):
                with self.assertRaises(ValueError):
                    download_dataset(output=target)
            self.assertFalse(target.exists())
            self.assertFalse((root / "escape.txt").exists())

            failure = subprocess.CalledProcessError(1, "kaggle")
            with patch("bilstm_pipeline.download.shutil.which", return_value="kaggle"), patch("bilstm_pipeline.download.subprocess.run", side_effect=failure):
                with self.assertRaises(subprocess.CalledProcessError):
                    download_dataset(output=target)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
