"""Track isolated runs. / Registra execuções isoladas."""

import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import subprocess
import tomllib
from uuid import uuid4

import numpy as np
import tomli_w

from .configuration import load_config, project_root
from .plots import plot_demo
from .synthetic import run_components

_RUN_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")
_MANIFEST_FILENAME = "manifest.json"


def run_path(run_id: str) -> Path:
    """Resolve a safe run ID. / Resolve um ID de execução seguro."""
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("Invalid run ID: use letters, digits, hyphens or underscores.")
    return project_root() / "output" / run_id


def write_json(path: Path, data: dict) -> None:
    """Write UTF-8 JSON. / Grava JSON em UTF-8."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _environment() -> dict:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root(), check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=project_root(), check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    return {
        "python": platform.python_version(), "system": platform.system(),
        "architecture": platform.machine(), "git_revision": revision, "uncommitted_changes": dirty,
        "packages": {name: version(name) for name in ("numpy", "matplotlib", "bilstm-experimentos")},
    }


def run(config_path: str | Path, overrides: dict | None = None, *, run_id: str | None = None) -> str:
    """Run the synthetic demo. / Executa a demonstração sintética."""
    config = load_config(config_path, overrides)
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    target = run_path(run_id)
    target.mkdir(parents=True, exist_ok=False)
    manifest = {
        "id": run_id, "status": "running", "type": "synthetic_demo",
        "origin": "Synthetic demonstration data; not scientific results.",
        "unit": "illustrative observation", "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        write_json(target / _MANIFEST_FILENAME, manifest)
        with (target / "effective_config.toml").open("wb") as stream:
            tomli_w.dump(config, stream)
        write_json(target / "environment.json", _environment())
        values = np.random.default_rng(config["experiment"]["seed"]).normal(size=config["experiment"]["points"])
        with (target / "synthetic_data.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["observation", "illustrative_value"])
            writer.writerows(enumerate(values, start=1))
        plot_demo(values, target / "figures")
        run_components(target, seed=config["experiment"]["seed"])
        manifest.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat())
        (target / "run.log").write_text("Synthetic demonstration completed.\n", encoding="utf-8")
        write_json(target / _MANIFEST_FILENAME, manifest)
    except Exception as error:
        manifest.update(status="failed", finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(target / _MANIFEST_FILENAME, manifest)
        (target / "run.log").write_text(f"Failure: {type(error).__name__}.\n", encoding="utf-8")
        raise
    return run_id


def load_run(run_id: str) -> dict:
    """Read a completed run, including legacy runs. / Lê execução concluída, inclusive anterior."""
    target = run_path(run_id)
    manifest_path = target / _MANIFEST_FILENAME
    legacy = not manifest_path.exists()
    if legacy:
        manifest_path = target / "manifesto.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = manifest.get("estado") if legacy else manifest.get("status")
    if status not in ("completed", "concluida"):
        raise ValueError(f"Run {run_id} is not complete: {status}.")
    config_path = target / ("config_efetiva.toml" if legacy else "effective_config.toml")
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    return {"manifest": manifest, "config": config, "directory": target, "legacy": legacy}
