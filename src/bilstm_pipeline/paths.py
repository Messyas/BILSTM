"""Project path utilities. / Utilitarios de caminhos do projeto."""

from pathlib import Path


def project_root() -> Path:
    """Find the project root. / Localiza a raiz do projeto."""
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for directory in (start, *start.parents):
            if (directory / "pyproject.toml").is_file() and (directory / "configs").is_dir():
                return directory
    raise FileNotFoundError("Cannot locate the BILSTM project root.")
