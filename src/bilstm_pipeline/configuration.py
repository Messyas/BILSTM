"""Load demonstration settings. / Carrega configurações da demonstração."""

from pathlib import Path
import tomllib


def project_root() -> Path:
    """Find the project root. / Localiza a raiz do projeto."""
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for directory in (start, *start.parents):
            if (directory / "pyproject.toml").is_file() and (directory / "configs").is_dir():
                return directory
    raise FileNotFoundError("Cannot locate the BILSTM project root.")


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    """Validate demonstration parameters. / Valida parâmetros da demonstração."""
    path = Path(path)
    if not path.is_absolute():
        path = project_root() / path
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    if set(config) != {"experiment"} or not isinstance(config["experiment"], dict):
        raise ValueError("Configuration must contain only an [experiment] table.")
    values = {"type": "synthetic_demo", "seed": 42, "points": 12}
    unknown = set(config["experiment"]) - set(values)
    if unknown:
        raise ValueError(f"Unknown parameters: {', '.join(sorted(unknown))}")
    values.update(config["experiment"])
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict):
        raise ValueError("Overrides must be a dictionary.")
    unknown = set(overrides) - {"seed", "points"}
    if unknown:
        raise ValueError(f"Unknown overrides: {', '.join(sorted(unknown))}")
    values.update(overrides)
    if values["type"] != "synthetic_demo":
        raise ValueError("Only the 'synthetic_demo' experiment is implemented.")
    if type(values["seed"]) is not int or not 0 <= values["seed"] < 2**32:
        raise ValueError("seed must be an integer between 0 and 2^32 - 1.")
    if type(values["points"]) is not int or not 2 <= values["points"] <= 1000:
        raise ValueError("points must be an integer between 2 and 1000.")
    return {"experiment": values}
