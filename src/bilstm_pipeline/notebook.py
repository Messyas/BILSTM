"""Notebook facade for real research workflows. / Interface de notebooks para fluxos reais de pesquisa."""

_EVALUATION_FILENAME = "evaluation.json"

def prepare_research(config: str):
    """Prepare the real dataset manifest. / Prepara o manifesto do dataset real."""
    from .research_data import prepare_dataset
    return prepare_dataset(config)


def show_research_dataset(config: str) -> None:
    """Display real split counts. / Exibe contagens das divisões reais."""
    import pandas as pd
    from IPython.display import display
    from .research_config import load_research_config, resolve_project_path
    from .research_data import load_manifest
    settings = load_research_config(config)
    manifest = load_manifest(resolve_project_path(settings["data"]["manifest"]))
    print(f"Dataset: {manifest['dataset_name']} — source: {manifest['source']}")
    display(pd.DataFrame(manifest["split_summary"]).T)
    display(manifest["split_metadata"])


def train_research(config: str, *, seed: int | None = None, device: str | None = None) -> str:
    """Train the proposed model. / Treina o modelo proposto."""
    from .research_training import train_model
    return train_model(config, seed=seed, device=device)


def show_research_training(run_id: str) -> None:
    """Display training history. / Exibe o historico de treinamento."""
    import json
    import pandas as pd
    import tomllib
    from IPython.display import Image, display
    from .execution import run_path
    directory = run_path(run_id)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("type") != "proposed_reconstruction":
        raise ValueError("Expected a completed proposed reconstruction run.")
    with (directory / "effective_config.toml").open("rb") as stream:
        config = tomllib.load(stream)
    history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
    print(f"Run ID: {run_id} - proposed reconstruction, not the original checkpoint.")
    display(config)
    display(pd.DataFrame(history["epochs"]))
    display(Image(filename=str(directory / "figures" / "training_history.png")))


def evaluate_research(run_id: str, *, device: str | None = None, robustness: bool = False, video_compression: bool = False):
    """Evaluate a held-out test split. / Avalia a partição de teste reservada."""
    from .research_evaluation import evaluate_model
    return evaluate_model(run_id, device=device, robustness=robustness, video_compression=video_compression)


def evaluate_external(run_id: str, external_config: str, *, device: str | None = None, robustness: bool = False, video_compression: bool = False):
    """Evaluate an independent dataset. / Avalia um dataset independente."""
    from .research_config import load_research_config
    from .research_evaluation import evaluate_model
    config = load_research_config(external_config)
    return evaluate_model(run_id, dataset_root=config["data"]["root"], external_manifest=config["data"]["manifest"], device=device, robustness=robustness, video_compression=video_compression)


def show_research_evaluation(path) -> None:
    """Display evaluation outputs. / Exibe resultados da avaliação."""
    import json
    from pathlib import Path
    import pandas as pd
    from IPython.display import Image, display
    target = Path(path)
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Evaluation is not complete.")
    evaluation = json.loads((target / "clean" / _EVALUATION_FILENAME).read_text(encoding="utf-8"))
    intervals = json.loads((target / "confidence_intervals.json").read_text(encoding="utf-8"))
    timing = json.loads((target / "timing.json").read_text(encoding="utf-8"))
    display(manifest)
    display(pd.DataFrame(evaluation["metrics"]["classes"]).T)
    display(pd.DataFrame([evaluation["metrics"]], columns=["total", "accuracy", "specificity", "false_positive_rate", "f1_macro", "f1_weighted"]))
    display(pd.DataFrame(intervals).T)
    display(timing)
    display(Image(filename=str(target / "figures" / "confusion_matrix.png")))
    display(Image(filename=str(target / "figures" / "binary_metrics.png")))
    if manifest.get("robustness"):
        comparisons = json.loads((target / "robustness.json").read_text(encoding="utf-8"))
        rows = []
        for condition, result in comparisons["comparisons"].items():
            interval = result["paired_accuracy_difference"]
            rows.append({"condition": condition, "accuracy": result["metrics"]["accuracy"], "difference": interval["estimate"], "lower": interval["lower"], "upper": interval["upper"], "skipped_resamples": interval["skipped_resamples"]})
        display(comparisons["parameters"])
        display(pd.DataFrame(rows))
        display(Image(filename=str(target / "figures" / "robustness.png")))
