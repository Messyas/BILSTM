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

from .paths import project_root

DATASET_REF = "mohamedmustafa/real-life-violence-situations-dataset"
RWF2000_DATASET_REF = "vulamnguyen/rwf2000"
SCVD_DATASET_REF = "toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd"
HOCKEY_DATASET_REF = "yassershrief/hockey-fight-vidoes"
DATASET_ALIASES = {
    "rlvs": DATASET_REF,
    "rwf2000": RWF2000_DATASET_REF,
    "scvd": SCVD_DATASET_REF,
    "hockey": HOCKEY_DATASET_REF,
}
DEFAULT_VERSION = 1
METADATA_FILENAME = "download_metadata.json"


def _resolve_dataset_ref(dataset: str) -> tuple[str, str]:
    """Resolve alias or Kaggle owner/slug reference. / Resolve apelido ou referência owner/slug do Kaggle."""
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("Dataset reference must be a nonempty string.")
    token = dataset.strip()
    alias = token.casefold()
    if alias in DATASET_ALIASES:
        return DATASET_ALIASES[alias], alias
    parts = token.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Dataset must be a known alias ({', '.join(DATASET_ALIASES)}) or '<owner>/<slug>'.")
    reverse = {ref: key for key, ref in DATASET_ALIASES.items()}
    return token, reverse.get(token, parts[1])


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


def download_dataset(*, dataset: str = DATASET_REF, version: int = DEFAULT_VERSION, output: str | Path | None = None) -> Path:
    """Download and extract one version. / Baixa e extrai uma versão."""
    if type(version) is not int or version < 1:
        raise ValueError("Dataset version must be a positive integer.")
    dataset_ref, short_name = _resolve_dataset_ref(dataset)
    target = Path(output).expanduser() if output is not None else project_root() / "data" / "raw" / f"{short_name}-v{version}"
    target = target.resolve()
    if target.exists():
        raise FileExistsError(f"Destination already exists: {target}. Choose another path.")
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise FileNotFoundError("Kaggle CLI not found. Run this script with 'uv run --group data'.")
    target.parent.mkdir(parents=True, exist_ok=True)
    reference = f"{dataset_ref}/{version}"
    with tempfile.TemporaryDirectory(prefix=f".{short_name}-download-", dir=target.parent) as temporary:
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
            "dataset": dataset_ref, "version": version,
            "url": f"https://www.kaggle.com/datasets/{dataset_ref}",
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "archive_sha256": checksum, "archive_bytes": archive.stat().st_size,
            "extracted_files": file_count,
        }
        (extracted / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if target.exists():
            raise FileExistsError(f"Destination already exists: {target}. Choose another path.")
        extracted.rename(target)
    return target
