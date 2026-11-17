# Resultados BiLSTM: protocolo fechado no Heisenberg (PR #1)

São os artefatos gerados no cluster Heisenberg (CNPEM) em 16/11/2026, a partir do branch `heisenberg-execucao`. Só esta pasta de `output/` é versionada; checkpoints, predições por vídeo e logs ficam no servidor. Para regerar: `uv run --no-sync python scripts/consolidate_results.py <3 run_ids>` e `scripts/summarize_seeds.py --main <3 run_ids>`.

## Runs principais (seed 42)

| backbone | run_id |
|---|---|
| mobilenet_v2 | rlvs-20261116T182754Z-9f8e5bbb |
| densenet121 | rlvs-20261116T182320Z-2c934236 |
| inception_v3 | rlvs-20261116T182320Z-598ec442 |

## Tabelas (geradas por `scripts/consolidate_results.py`)

- **`RESULTADOS.md`** é o relatório legível, com as seguintes seções:
  - protocolo;
  - treino e limiar;
  - teste RLVS com IC 95%;
  - RWF-2000;
  - matrizes de confusão;
  - robustez;
  - latência;
  - hardware;
  - proveniência.
- **`resumo.json`** é a fonte completa. Para cada backbone, traz:
  - `training`: épocas, tempo e hardware de treino;
  - `threshold`: limiar congelado e contagens na validação;
  - `test.metrics` e `test.intervals`: todas as métricas do teste RLVS, por classe e macro, com IC 95% por bootstrap de vídeos, 1000 reamostragens;
  - `test.robustness.comparisons.<condição>`: para cada condição, as métricas (`metrics`), os IC das 11 métricas (`confidence_intervals`) e a diferença pareada de acurácia contra o caso limpo (`paired_accuracy_difference`);
  - `external`: o mesmo conjunto para o RWF-2000;
  - `latency`: os tempos offline e em streaming, com mediana, p95, máximo e média, mais o hardware da medição.
- **CSVs:** `treino_e_limiar`, `teste_rlvs`, `externo_rwf2000`, `matrizes_confusao`, `robustez` (só o Δ de acurácia; os IC completos estão no JSON) e `latencia`.
- **`suplementar_seeds/`** (gerado por `scripts/summarize_seeds.py`) é o adendo de estabilidade:
  - `SEEDS.md` traz média ± desvio padrão, com mínimo e máximo, de 6 seeds (1–5 e 42) por backbone, em 18 runs com o mesmo split;
  - `por_run.csv` traz os valores de cada run.

## `figuras/<backbone>/` (copiadas por `consolidate_results.py`; PDF vetorial e PNG)

| arquivo | conteúdo |
|---|---|
| `treino_historico` | perda e acurácia de treino e validação por época |
| `teste_rlvs_matriz_confusao` | contagens no teste RLVS (402 vídeos) com o limiar congelado |
| `teste_rlvs_metricas` | precisão, recall, F1 e especificidade no teste RLVS |
| `teste_rlvs_robustez` | Δ de acurácia contra o caso limpo para cada perturbação, com IC 95% |
| `externo_rwf2000_matriz_confusao` | contagens no RWF-2000 (2000 vídeos) com o mesmo limiar |
| `externo_rwf2000_metricas` | precisão, recall, F1 e especificidade no RWF-2000 |

As figuras são as geradas automaticamente pelo pipeline. Elas ainda não foram preparadas para publicação:
- os títulos estão em inglês;
- a matriz de confusão tem o título cortado;
- nenhuma figura indica o backbone dentro da imagem.

Antes de entrar no capítulo, precisam ser refeitas no formato da editora (item E09 do backlog).

## Hardware

- **Treino:** work5, com RTX 4090 24 GB, Ryzen 7 7700X e 125,5 GiB de RAM.
- **Avaliação:** em CPU, no work7, com Ryzen 9 7950X.
- **Latência:** work6, com RTX 4090.

A ficha completa (driver, CUDA, cuDNN, núcleos físicos e lógicos) está em `RESULTADOS.md`, seção Hardware, e nos campos `hardware` do `resumo.json`.
