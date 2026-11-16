"""Consolidate the closed-protocol runs into chapter tables (E09). / Consolida os runs do protocolo fechado em tabelas (E09).

Uso: uv run --no-sync python scripts/consolidate_results.py RUN_ID [RUN_ID ...] [--out output/consolidado]
Lê só artefatos já gravados em output/<RUN_ID>/; não treina nem avalia nada.
"""

import argparse
import csv
import json
from pathlib import Path
import tomllib

from bilstm_pipeline.paths import project_root

METRICS = [
    ("accuracy", "Acurácia"),
    ("positive_precision", "Precisão (Violence)"),
    ("positive_recall", "Recall (Violence)"),
    ("positive_f1", "F1 (Violence)"),
    ("negative_precision", "Precisão (NonViolence)"),
    ("negative_recall", "Recall (NonViolence)"),
    ("negative_f1", "F1 (NonViolence)"),
    ("specificity", "Especificidade"),
    ("f1_macro", "F1 macro"),
    ("false_positive_rate", "FPR"),
]
CONDITIONS = [("brightness", "Brilho"), ("contrast", "Contraste"), ("noise", "Ruído gaussiano"), ("occlusion", "Oclusão central"), ("jpeg_frames", "JPEG por quadro"), ("h264_video", "Recodificação H.264")]


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluation(run_dir: Path, kind: str) -> Path:
    found = []
    for item in sorted((run_dir / "evaluations").glob("*/manifest.json")):
        status = _json(item)
        if status.get("status") == "completed" and status.get("type") == kind:
            found.append(item.parent)
    if not found:
        raise FileNotFoundError(f"No completed {kind} evaluation in {run_dir}.")
    if len(found) > 1:
        print(f"Aviso: {len(found)} avaliações '{kind}' em {run_dir.name}; usando a mais recente ({found[-1].name}).")
    return found[-1]


def _latency(run_dir: Path) -> dict:
    items = sorted((run_dir / "latency").glob("*/latency.json"))
    cuda = [_json(item) for item in items if _json(item)["device"].startswith("cuda")]
    if not cuda:
        raise FileNotFoundError(f"No GPU latency result in {run_dir}.")
    return cuda[-1]


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}".replace(".", ",")


def _ci(intervals: dict, key: str) -> tuple:
    item = intervals.get(key, {})
    return item.get("estimate"), item.get("lower"), item.get("upper")


def collect(run_id: str) -> dict:
    run_dir = project_root() / "output" / run_id
    status = _json(run_dir / "manifest.json")
    with (run_dir / "effective_config.toml").open("rb") as stream:
        config = tomllib.load(stream)
    history = _json(run_dir / "history.json")["epochs"]
    threshold = _json(run_dir / "threshold.json")
    test_dir, external_dir = _evaluation(run_dir, "test"), _evaluation(run_dir, "external")
    best_epoch = max(history, key=lambda row: (row["validation_accuracy"], -row["validation_loss"]))["epoch"]
    return {
        "run_id": run_id,
        "backbone": config["model"]["backbone"],
        "training": {"epochs_run": len(history), "best_epoch": best_epoch, "best_validation_accuracy": status["best_validation_accuracy"], "training_time_seconds": status["training_time_seconds"], "environment": _json(run_dir / "environment.json")},
        "threshold": threshold,
        "test": {"dir": str(test_dir.relative_to(project_root())), "metrics": _json(test_dir / "clean" / "evaluation.json")["metrics"], "intervals": _json(test_dir / "confidence_intervals.json"), "status": _json(test_dir / "manifest.json"), "robustness": _json(test_dir / "robustness.json")},
        "external": {"dir": str(external_dir.relative_to(project_root())), "metrics": _json(external_dir / "clean" / "evaluation.json")["metrics"], "intervals": _json(external_dir / "confidence_intervals.json"), "status": _json(external_dir / "manifest.json")},
        "latency": _latency(run_dir),
    }


def _metric_rows(runs, section):
    rows = []
    for key, label in METRICS:
        row = {"metrica": label}
        for run in runs:
            estimate, lower, upper = _ci(run[section]["intervals"], key)
            row[run["backbone"]] = f"{_fmt(estimate)} [{_fmt(lower)}; {_fmt(upper)}]"
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _md_table(rows: list[dict]) -> str:
    header = list(rows[0])
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(row[column]) for column in header) + " |" for row in rows]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--out", default="output/consolidado")
    args = parser.parse_args()
    runs = [collect(run_id) for run_id in args.run_ids]
    out = project_root() / args.out
    out.mkdir(parents=True, exist_ok=True)

    calibration = [{"backbone": run["backbone"], "run_id": run["run_id"], "epocas": run["training"]["epochs_run"], "melhor_epoca": run["training"]["best_epoch"], "acc_validacao": _fmt(run["training"]["best_validation_accuracy"]), "tempo_treino_s": _fmt(run["training"]["training_time_seconds"], 0), "limiar": _fmt(run["threshold"]["threshold"], 4), "fpr_validacao": _fmt(run["threshold"]["validation_fpr"]), "recall_validacao": _fmt(run["threshold"]["validation_recall"])} for run in runs]
    test_rows, external_rows = _metric_rows(runs, "test"), _metric_rows(runs, "external")
    matrices = [{"backbone": run["backbone"], "conjunto": name, "TN": m[0][0], "FP": m[0][1], "FN": m[1][0], "TP": m[1][1], "n": run[section]["metrics"]["total"]} for run in runs for section, name in (("test", "RLVS teste"), ("external", "RWF-2000")) for m in [run[section]["metrics"]["confusion_matrix"]]]
    robustness = []
    for key, label in CONDITIONS:
        row = {"condicao": label}
        for run in runs:
            comparison = run["test"]["robustness"]["comparisons"].get(key)
            if comparison is None:
                row[run["backbone"]] = "—"
                continue
            diff = comparison["paired_accuracy_difference"]
            row[run["backbone"]] = f"acc {_fmt(comparison['metrics']['accuracy'])} (Δ {_fmt(diff['estimate'])} [{_fmt(diff['lower'])}; {_fmt(diff['upper'])}]); FPR {_fmt(comparison['metrics']['false_positive_rate'])}"
        robustness.append(row)
    latency = []
    for run in runs:
        lat = run["latency"]
        off, stream = lat["offline"], lat["streaming"]
        latency.append({
            "backbone": run["backbone"],
            "offline_total_mediana_ms": _fmt(off["total"]["median_ms"], 1), "offline_total_p95_ms": _fmt(off["total"]["p95_ms"], 1),
            "offline_inferencia_mediana_ms": _fmt(off["inference"]["median_ms"], 2), "offline_inferencia_p95_ms": _fmt(off["inference"]["p95_ms"], 2),
            "stream_decisao_mediana_ms": _fmt(stream["per_decision_total"]["median_ms"], 2), "stream_decisao_p95_ms": _fmt(stream["per_decision_total"]["p95_ms"], 2),
            "stream_quadro_p95_ms": _fmt(stream["per_frame_decode"]["p95_ms"] + stream["per_frame_ingest_resize"]["p95_ms"], 2),
            "carga_tempo_real_p95": _fmt(stream["budget"]["compute_load_fraction_p95"], 3),
            "atraso_janela_s": _fmt(stream["budget"]["window_formation_delay_s"], 1),
            "videos": lat["videos"], "decisoes": stream["per_decision_total"]["n"],
        })
    hardware = runs[0]["latency"]["hardware"]
    eval_hardware = _json(project_root() / runs[0]["test"]["dir"] / "hardware.json")

    for name, rows in (("treino_e_limiar", calibration), ("teste_rlvs", test_rows), ("externo_rwf2000", external_rows), ("matrizes_confusao", matrices), ("robustez", robustness), ("latencia", latency)):
        _write_csv(out / f"{name}.csv", rows)
    gpu = hardware["gpu"]
    hw_text = (f"- **Treino e latência (GPU):** {hardware['hostname']} — {gpu['name']} ({gpu['memory_gib']} GiB, CC {gpu['compute_capability']}, driver {gpu['driver']}, CUDA {gpu['cuda_runtime']}, cuDNN {gpu['cudnn']}); "
               f"CPU {hardware['cpu']['model']} ({hardware['cpu']['physical_cores']} núcleos físicos / {hardware['cpu']['logical_cpus']} lógicos; {hardware['cpu']['cpus_allocated_to_process']} alocados ao job); RAM {hardware['ram_gib']} GiB; "
               f"{hardware['torch_threads']} threads PyTorch; {hardware['os']}.\n"
               f"- **Calibração, teste, robustez e RWF-2000 (CPU, mesmo dispositivo para todos os scores):** {eval_hardware['hostname']} — CPU {eval_hardware['cpu']['model']} ({eval_hardware['cpu']['physical_cores']} físicos / {eval_hardware['cpu']['logical_cpus']} lógicos; {eval_hardware['cpu']['cpus_allocated_to_process']} alocados); RAM {eval_hardware['ram_gib']} GiB; {eval_hardware['torch_threads']} threads PyTorch.\n"
               f"- **Software:** PyTorch {runs[0]['training']['environment']['torch']}, torchvision {runs[0]['training']['environment']['torchvision']}, OpenCV {runs[0]['training']['environment']['opencv']}, Python {runs[0]['training']['environment']['python']}; FFmpeg: {runs[0]['test']['status'].get('ffmpeg_version')}.")
    stream_protocol = runs[0]["latency"]["streaming"]["protocol"]
    report = f"""# Resultados consolidados — protocolo fechado (RLVS → RWF-2000)

Gerado por `scripts/consolidate_results.py` a partir dos artefatos em `output/<RUN_ID>/`. Retreinamento em PyTorch a partir do ImageNet; **não** é o checkpoint `.keras` original.

## Protocolo

- 3 backbones (`mobilenet_v2`, `densenet121`, `inception_v3`) + BiLSTM + atenção; 15 quadros de 100 × 100; split por vídeo 0,64 / 0,16 / 0,20, seed 42 (1277 / 319 / 402 vídeos; 1998 após excluir um par idêntico com rótulos opostos).
- Limiar calibrado **na validação do RLVS** com FPR ≤ {_fmt(runs[0]['threshold']['target_fpr'], 2)} (maximizando o recall) e congelado para teste, robustez e RWF-2000. A validação também escolheu o `best.pt`.
- IC 95%: bootstrap percentil reamostrando **vídeos** (grupos de conteúdo idêntico juntos), 1000 reamostragens. Classe positiva: Violence/Fight.

## Treino e limiar congelado

{_md_table(calibration)}

## Teste no RLVS (n = {runs[0]['test']['metrics']['total']} vídeos) — estimativa [IC 95%]

{_md_table(test_rows)}

## Validação externa no RWF-2000 (n = {runs[0]['external']['metrics']['total']} vídeos, limiar congelado do RLVS, sem ajuste)

{_md_table(external_rows)}

## Matrizes de confusão

{_md_table(matrices)}

## Robustez sintética no teste do RLVS (Δ = diferença pareada de acurácia vs. limpo, IC 95%)

{_md_table(robustness)}

Perturbações sintéticas sobre vídeos observados; não representam câmeras ou aglomerações reais. JPEG por quadro e recodificação H.264 (libx264, CRF {runs[0]['test']['robustness']['parameters']['video_crf']}) são condições distintas. ICs completos de todas as métricas por condição estão em `robustness.json` de cada avaliação.

## Latência (RTX 4090, lote 1, {WARMUP} inferências de aquecimento)

{_md_table(latency)}

- **Offline:** clipe gravado inteiro — decodificação em duas passagens + pré-processamento + inferência. Não é atraso de alerta ao vivo.
- **Streaming (tempo real simulado):** cada vídeo reproduzido a {_fmt(stream_protocol['fps'], 0)} fps, janela de {_fmt(stream_protocol['window_seconds'], 1)} s ({stream_protocol['window_frames']} quadros, 15 amostrados), decisão a cada {_fmt(stream_protocol['stride_seconds'], 1)} s. O alerta só pode sair depois de a janela se completar (atraso de formação = {_fmt(stream_protocol['window_seconds'], 1)} s), mais o tempo por decisão. "Carga" = fração do intervalo entre decisões gasta com processamento (p95). Só o tempo é medido; a acurácia em janelas deslizantes não foi avaliada, e o modelo foi treinado com amostragem do clipe inteiro. O tempo de decodificação por quadro é amortizado (decodificador multithread do OpenCV) e não inclui captura/rede de uma câmera real.

## Hardware

{hw_text}

## Proveniência

""" + "\n".join(f"- `{run['backbone']}`: run `{run['run_id']}`; teste `{run['test']['dir']}`; RWF `{run['external']['dir']}`; limiar `output/{run['run_id']}/threshold.json`." for run in runs) + "\n"
    report = report.replace("{WARMUP}", str(runs[0]["latency"]["warmup_inferences"]))
    (out / "RESULTADOS.md").write_text(report, encoding="utf-8")
    (out / "resumo.json").write_text(json.dumps(runs, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
