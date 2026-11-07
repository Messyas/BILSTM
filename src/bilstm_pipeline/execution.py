"""Manage research run paths and JSON artifacts. / Gerencia caminhos e artefatos JSON das execucoes."""

import json
from pathlib import Path
import re

from .paths import project_root

_RUN_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")


def run_path(run_id: str) -> Path:
    """Resolve a safe run ID. / Resolve um ID de execucao seguro."""
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("Invalid run ID: use letters, digits, hyphens or underscores.")
    return project_root() / "output" / run_id


def write_json(path: Path, data: dict) -> None:
    """Write UTF-8 JSON. / Grava JSON em UTF-8."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
