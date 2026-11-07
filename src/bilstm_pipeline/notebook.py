"""Notebook facade. / Interface para notebooks."""

from .execution import load_run, run


def show_parameters(run_id: str) -> None:
    """Display effective parameters. / Exibe parâmetros efetivos."""
    from IPython.display import display
    result = load_run(run_id)
    print(f"Run: {run_id}")
    display(result["config"])


def show_results(run_id: str) -> None:
    """Display a completed demonstration. / Exibe demonstração concluída."""
    from IPython.display import Image, display
    import pandas as pd
    result = load_run(run_id)
    print(result["manifest"].get("origem") if result["legacy"] else result["manifest"]["origin"])
    directory = result["directory"]
    data = "dados_sinteticos.csv" if result["legacy"] else "synthetic_data.csv"
    image = "figuras/demonstracao.png" if result["legacy"] else "figures/demo.png"
    display(pd.read_csv(directory / data))
    display(Image(filename=str(directory / image)))
    if not result["legacy"] and (directory / "evaluation" / "evaluation.json").exists():
        import json
        evaluation = json.loads((directory / "evaluation" / "evaluation.json").read_text(encoding="utf-8"))
        print("Synthetic classification example — not scientific results")
        display(pd.DataFrame(evaluation["metrics"]["classes"]).T)
        display(Image(filename=str(directory / "figures" / "confusion_matrix.png")))
        display(Image(filename=str(directory / "figures" / "binary_metrics.png")))
