"""Train the proposed reconstruction. / Treina a reconstrução proposta."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import random
import shutil
import subprocess
from uuid import uuid4
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import tomli_w

from .execution import run_path, write_json
from .plots import plot_training_history
from .research_config import load_research_config, resolve_project_path
from .research_data import _hash_file, decode_video, load_manifest, video_path
from .research_model import ConflictModel

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_RUN_MANIFEST_FILENAME = "manifest.json"


def select_device(name: str) -> torch.device:
    """Select and validate the device. / Seleciona e valida o dispositivo."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a compatible PyTorch build or explicitly select cpu.")
    return torch.device(name)


def set_seed(seed: int) -> None:
    """Set common random seeds. / Define sementes aleatórias comuns."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def video_tensor(frames: np.ndarray) -> torch.Tensor:
    """Normalize RGB frames. / Normaliza quadros RGB."""
    normalized = (frames - _MEAN) / _STD
    return torch.from_numpy(normalized.transpose(0, 3, 1, 2).copy())


class VideoDataset(Dataset):
    """Read one sequence per video. / Lê uma sequência por vídeo."""

    def __init__(self, root: Path, records: list[dict], video_config: dict):
        self.root, self.records, self.video_config = root, records, video_config

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        frames = decode_video(video_path(self.root, record), **self.video_config)
        return video_tensor(frames), int(record["label"])


def _device_metadata(device: torch.device) -> dict:
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True, timeout=5).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    return {"framework": "pytorch", "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "torchvision": version("torchvision"), "opencv": version("opencv-python-headless"), "cuda_runtime": getattr(getattr(torch, "version", None), "cuda", None), "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "git_revision": revision, "uncommitted_changes": dirty}


def _run_epoch(model, loader, device, loss_fn, optimizer=None) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss, total_samples = 0.0, 0
    with torch.set_grad_enabled(is_training):
        for videos, labels in loader:
            videos, labels = videos.to(device), labels.to(device)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(videos), labels)
            if is_training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total_samples += len(labels)
    if not total_samples:
        raise ValueError("Training and validation splits must not be empty.")
    return total_loss / total_samples


def _training_records(root: Path, manifest_file: Path) -> tuple[dict, list[dict], list[dict]]:
    """Load verified training records. / Carrega registros de treino verificados."""
    manifest = load_manifest(manifest_file)
    records_by_split = {
        name: [record for record in manifest["records"] if manifest["assignments"][record["sample_id"]] == name]
        for name in ("train", "validation")
    }
    for record in manifest["records"]:
        video_path(root, record, verify_hash=True)
    for name, records in records_by_split.items():
        if {record["label"] for record in records} != {0, 1}:
            raise ValueError(f"Split {name} must contain both classes.")
    return manifest, records_by_split["train"], records_by_split["validation"]


def _write_initial_artifacts(target: Path, status: dict, config: dict, manifest_file: Path, device: torch.device) -> None:
    """Persist run inputs and environment. / Salva entradas e ambiente da execucao."""
    write_json(target / _RUN_MANIFEST_FILENAME, status)
    with (target / "effective_config.toml").open("wb") as stream:
        tomli_w.dump(config, stream)
    shutil.copyfile(manifest_file, target / "dataset_manifest.json")
    write_json(target / "environment.json", _device_metadata(device))


def _create_loaders(root: Path, config: dict, train_records: list[dict], val_records: list[dict]):
    """Build deterministic data loaders. / Cria carregadores de dados deterministas."""
    generator = torch.Generator().manual_seed(config["training"]["seed"])
    options = {"batch_size": config["training"]["batch_size"], "num_workers": config["training"]["num_workers"]}
    train_loader = DataLoader(VideoDataset(root, train_records, config["video"]), shuffle=True, generator=generator, **options)
    val_loader = DataLoader(VideoDataset(root, val_records, config["video"]), shuffle=False, **options)
    return train_loader, val_loader


def _train_epochs(model, train_loader, val_loader, device: torch.device, config: dict, target: Path) -> tuple[list[dict], float]:
    """Train and checkpoint the best validation epoch. / Treina e salva a melhor epoca de validacao."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    loss_fn = nn.CrossEntropyLoss()
    history, best_loss = [], float("inf")
    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss = _run_epoch(model, train_loader, device, loss_fn, optimizer)
        val_loss = _run_epoch(model, val_loader, device, loss_fn)
        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            raise ValueError("Training or validation loss is not finite.")
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        write_json(target / "history.json", {"epochs": history})
        line = f"Epoch {epoch}: train_loss={train_loss:.6f}, validation_loss={val_loss:.6f}"
        print(line, flush=True)
        with (target / "training.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "validation_loss": val_loss}, target / "best.pt")
    return history, best_loss


def _finish_run(target: Path, status: dict, *, best_loss: float | None = None, error: Exception | None = None) -> None:
    """Record run completion or failure. / Registra conclusao ou falha da execucao."""
    status.update(status="failed" if error else "completed", finished_utc=datetime.now(timezone.utc).isoformat())
    if error is not None:
        status["error_type"] = type(error).__name__
    if best_loss is not None:
        status["best_validation_loss"] = best_loss
    write_json(target / _RUN_MANIFEST_FILENAME, status)
    if error is not None:
        with (target / "training.log").open("a", encoding="utf-8") as stream:
            stream.write(f"Failed: {type(error).__name__}: {error}\n")


def train_model(config_path: str | Path, *, dataset_root: str | Path | None = None, manifest_path: str | Path | None = None, seed: int | None = None, device: str | None = None) -> str:
    """Train using training and validation only. / Treina usando apenas treino e validacao."""
    config = load_research_config(config_path, seed=seed)
    if device is not None:
        if device not in ("cuda", "cpu", "auto"):
            raise ValueError("device must be cuda, cpu or auto.")
        config["training"]["device"] = device
    root = resolve_project_path(dataset_root or config["data"]["root"])
    manifest_file = resolve_project_path(manifest_path or config["data"]["manifest"])
    config["data"].update(root=str(root), manifest=str(manifest_file))
    manifest, train_records, val_records = _training_records(root, manifest_file)
    selected = select_device(config["training"]["device"])
    set_seed(config["training"]["seed"])
    run_id = "rlvs-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    target = run_path(run_id)
    target.mkdir(parents=True, exist_ok=False)
    print(f"Run ID: {run_id}", flush=True)
    status: dict[str, Any] = {"id": run_id, "status": "running", "type": "proposed_reconstruction", "dataset": manifest["dataset_name"], "dataset_manifest_sha256": _hash_file(manifest_file), "started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        _write_initial_artifacts(target, status, config, manifest_file, selected)
        model = ConflictModel(**config["model"]).to(selected)
        train_loader, val_loader = _create_loaders(root, config, train_records, val_records)
        history, best_loss = _train_epochs(model, train_loader, val_loader, selected, config, target)
        plot_training_history(history, target / "figures")
        _finish_run(target, status, best_loss=best_loss)
    except Exception as error:
        _finish_run(target, status, error=error)
        raise
    return run_id
