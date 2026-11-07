# Notebooks

Os notebooks sao a interface das etapas reais: explicam cada etapa, indicam a configuracao e o ID da execucao, chamam a API do projeto e exibem tabelas e graficos. Nao ha notebooks de avaliacao com dados inventados.

Use ingles nos nomes, identificadores e textos dos notebooks; comentarios em codigo devem estar em ingles e portugues. Mantenha configuracoes de ambiente, processamento, treino, metricas, graficos e imports cientificos em `src/bilstm_pipeline/`. Cada notebook importa apenas a API de alto nivel.

Abra com `uv run --group notebooks jupyter lab` na raiz e selecione `Python (BILSTM Experimentos)`. Treino e avaliacao devem ocorrer no servidor preparado para PyTorch e GPU. Siga o [guia de execucao](../docs/plano-de-implementacao/EXECUCAO_NO_SERVIDOR.md).

| Notebook | Etapa |
| --- | --- |
| `01_prepare_dataset.ipynb` | Inventariar o RLVS, definir grupos e revisar particoes. |
| `02_train_baseline.ipynb` | Treinar a reconstrucao proposta no servidor. |
| `03_evaluate_test.ipynb` | Avaliar teste reservado e robustez. |
| `04_evaluate_external.ipynb` | Avaliar uma base externa preparada separadamente. |

A arquitetura e uma reconstrucao proposta e depende de comparacao com o modelo original. A existencia dos notebooks nao significa que treino ou avaliacao reais ja foram executados.
