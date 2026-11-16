"""Supplementary table: variation across training seeds. / Tabela suplementar: variação entre seeds de treino.

Uso: uv run --no-sync python scripts/summarize_seeds.py --main RUN_ID [RUN_ID ...] [--out output/consolidado/suplementar_seeds]
Junta os runs principais (seed 42) com os do adendo listados em output/_seeds/task_*.tsv.
Só o training.seed varia; o split (manifesto) e o conjunto de teste são os mesmos em todas as seeds.
"""

import argparse
import csv
import json
from pathlib import Path
import statistics
import tomllib

from bilstm_pipeline.paths import project_root

METRICS = [("accuracy", "Acurácia"), ("positive_recall", "Recall (Violence)"), ("positive_precision", "Precisão (Violence)"), ("specificity", "Especificidade"), ("f1_macro", "F1 macro"), ("false_positive_rate", "FPR")]
CONDITIONS = [("brightness", "Brilho"), ("contrast", "Contraste"), ("noise", "Ruído gaussiano"), ("occlusion", "Oclusão central"), ("jpeg_frames", "JPEG por quadro"), ("h264_video", "Recodificação H.264")]


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _latest(run_dir: Path, kind: str) -> Path | None:
    found = [item.parent for item in sorted((run_dir / "evaluations").glob("*/manifest.json")) if _json(item).get("status") == "completed" and _json(item).get("type") == kind]
    return found[-1] if found else None


def _run(run_id: str) -> dict | None:
    run_dir = project_root() / "output" / run_id
    with (run_dir / "effective_config.toml").open("rb") as stream:
        config = tomllib.load(stream)
    test, external = _latest(run_dir, "test"), _latest(run_dir, "external")
    if test is None or external is None or not (run_dir / "threshold.json").is_file():
        print(f"Aviso: {run_id} incompleto (sem limiar, teste ou RWF); ignorado.")
        return None
    def flat(metrics):
        return {"accuracy": metrics["accuracy"], "positive_recall": metrics["classes"]["1"]["recall"], "positive_precision": metrics["classes"]["1"]["precision"], "specificity": metrics["specificity"], "f1_macro": metrics["f1_macro"], "false_positive_rate": metrics["false_positive_rate"]}
    robustness = _json(test / "robustness.json")["comparisons"]
    return {
        "run_id": run_id, "backbone": config["model"]["backbone"], "seed": config["training"]["seed"],
        "best_validation_accuracy": _json(run_dir / "manifest.json")["best_validation_accuracy"],
        "threshold": _json(run_dir / "threshold.json")["threshold"],
        "test": flat(_json(test / "clean" / "evaluation.json")["metrics"]),
        "external": flat(_json(external / "clean" / "evaluation.json")["metrics"]),
        "robustness_delta": {key: robustness[key]["paired_accuracy_difference"]["estimate"] for key, _ in CONDITIONS if key in robustness},
    }


def _summary(values: list[float]) -> str:
    if not values:
        return "—"
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    fmt = lambda value: f"{value:.3f}".replace(".", ",")
    return f"{fmt(statistics.mean(values))} ± {fmt(sd)} ({fmt(min(values))}–{fmt(max(values))})"


def _md(rows):
    header = list(rows[0])
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"] + ["| " + " | ".join(str(row[key]) for key in header) + " |" for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--main", nargs="+", required=True, help="RUN_IDs principais (seed 42)")
    parser.add_argument("--out", default="output/consolidado/suplementar_seeds")
    args = parser.parse_args()
    ids = list(args.main) + [line.split("\t")[2].strip() for path in sorted((project_root() / "output" / "_seeds").glob("task_*.tsv")) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    runs = [run for run in (_run(run_id) for run_id in ids) if run is not None]
    backbones = sorted({run["backbone"] for run in runs})
    by = {bb: sorted((run for run in runs if run["backbone"] == bb), key=lambda run: run["seed"]) for bb in backbones}
    out = project_root() / args.out
    out.mkdir(parents=True, exist_ok=True)

    per_run = [{"backbone": run["backbone"], "seed": run["seed"], "run_id": run["run_id"], "acc_validacao": round(run["best_validation_accuracy"], 4), "limiar": round(run["threshold"], 4), **{f"teste_{key}": round(run["test"][key], 4) for key, _ in METRICS}, **{f"rwf_{key}": round(run["external"][key], 4) for key, _ in METRICS}, **{f"delta_{key}": round(value, 4) for key, value in run["robustness_delta"].items()}} for run in runs]
    with (out / "por_run.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_run[0]))
        writer.writeheader()
        writer.writerows(per_run)

    def table(getter, labels):
        rows = []
        for key, label in labels:
            row = {"metrica": label}
            for bb in backbones:
                row[f"{bb} (n={len(by[bb])})"] = _summary([value for value in (getter(run, key) for run in by[bb]) if value is not None])
            rows.append(row)
        return rows

    header_rows = table(lambda run, key: run[key], [("best_validation_accuracy", "Acurácia de validação (best.pt)"), ("threshold", "Limiar congelado")])
    test_rows = table(lambda run, key: run["test"][key], METRICS)
    external_rows = table(lambda run, key: run["external"][key], METRICS)
    robust_rows = table(lambda run, key: run["robustness_delta"].get(key), CONDITIONS)
    seeds_text = ", ".join(f"`{bb}`: {', '.join(str(run['seed']) for run in by[bb])}" for bb in backbones)
    report = f"""# Material suplementar — variação entre seeds de treino

Média ± desvio padrão amostral (mínimo–máximo) entre seeds de treino. Seeds: {seeds_text}. Os resultados principais do capítulo usam a seed 42 (protocolo fechado); esta tabela é um adendo.

**O que varia:** apenas `training.seed` (inicialização das camadas novas, ordem dos lotes, dropout e, na robustez, a semente do ruído gaussiano). **O que é fixo:** o split por vídeo (manifesto da seed 42), o conjunto de teste (402 vídeos), o RWF-2000 (2000 vídeos), a meta de FPR na validação e todo o protocolo. Para cada seed, o limiar foi recalibrado na validação e congelado antes do teste. A dispersão mede, portanto, a instabilidade do treino, não a variação amostral do teste, que é coberta pelos IC por bootstrap da tabela principal.

## Treino e limiar

{_md(header_rows)}

## Teste no RLVS

{_md(test_rows)}

## Validação externa no RWF-2000 (limiar congelado de cada seed)

{_md(external_rows)}

## Robustez: Δ acurácia vs. limpo (teste RLVS)

{_md(robust_rows)}

Valores por run em `por_run.csv`.
"""
    (out / "SEEDS.md").write_text(report, encoding="utf-8")
    print(f"Output: {out} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
