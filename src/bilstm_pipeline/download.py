"""Download the RLVS dataset. / Baixa o dataset RLVS."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
import zipfile

from .configuration import project_root

DATASET_REF = "mohamedmustafa/real-life-violence-situations-dataset"
DEFAULT_VERSION = 1
METADATA_FILENAME = "download_metadata.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_archive(archive: Path, destination: Path) -> int:
    """Extract safe regular files. / Extrai arquivos regulares seguros."""
    count = 0
    seen_files: set[str] = set()
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            name = info.filename
            relative = PurePosixPath(name)
            if (not name or "\\" in name or relative.is_absolute() or not relative.parts
                    or any(part in (".", "..") or ":" in part for part in relative.parts)
                    or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError(f"Unsafe archive member: {name!r}")
            target = destination.joinpath(*relative.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            key = "/".join(relative.parts).casefold()
            if key in seen_files or key == METADATA_FILENAME.casefold():
                raise ValueError(f"Duplicate or reserved archive member: {name!r}")
            seen_files.add(key)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(info) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            count += 1
    if not count:
        raise ValueError("The downloaded archive contains no files.")
    return count


def download_dataset(*, version: int = DEFAULT_VERSION, output: str | Path | None = None) -> Path:
    """Download and extract one version. / Baixa e extrai uma versão."""
    if type(version) is not int or version < 1:
        raise ValueError("Dataset version must be a positive integer.")
    target = Path(output).expanduser() if output is not None else project_root() / "data" / "raw" / f"rlvs-v{version}"
    target = target.resolve()
    if target.exists():
        raise FileExistsError(f"Destination already exists: {target}. Choose another path.")
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise FileNotFoundError("Kaggle CLI not found. Run this script with 'uv run --group data'.")
    target.parent.mkdir(parents=True, exist_ok=True)
    reference = f"{DATASET_REF}/{version}"
    with tempfile.TemporaryDirectory(prefix=".rlvs-download-", dir=target.parent) as temporary:
        staging = Path(temporary)
        archive_dir = staging / "archive"
        extracted = staging / "extracted"
        archive_dir.mkdir()
        extracted.mkdir()
        print(f"Downloading {reference} from Kaggle...")
        subprocess.run([kaggle, "datasets", "download", reference, "--path", str(archive_dir)], check=True)
        archives = list(archive_dir.glob("*.zip"))
        if len(archives) != 1:
            raise ValueError(f"Expected one ZIP archive from Kaggle, found {len(archives)}.")
        print("Calculating archive SHA-256 and extracting files...")
        archive = archives[0]
        checksum = _sha256(archive)
        file_count = _extract_archive(archive, extracted)
        metadata = {
            "dataset": DATASET_REF, "version": version,
            "url": f"https://www.kaggle.com/datasets/{DATASET_REF}",
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "archive_sha256": checksum, "archive_bytes": archive.stat().st_size,
            "extracted_files": file_count,
        }
        (extracted / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if target.exists():
            raise FileExistsError(f"Destination already exists: {target}. Choose another path.")
        extracted.rename(target)
    return target
