# Pipeline Python

Este diretório contém entradas finas de linha de comando. Elas interpretam argumentos e chamam a mesma API Python de alto nível usada pelos notebooks. A implementação compartilhada fica em `src/bilstm_pipeline/`; não duplique lógica de processamento ou gráficos nos scripts.

Etapas previstas pelo plano de experimentos: preparação e split dos dados por vídeo/cena; treino e validação; avaliação com métricas expandidas e intervalos de confiança; testes de robustez; e avaliação externa (*cross-dataset*). A implementação deve vir do código recebido e das decisões documentadas, sem duplicar a lógica em notebooks.

`experiment.py` executa e consulta uma demonstração com dados sintéticos. Os componentes reutilizáveis de métricas, divisão e robustez estão disponíveis; treino e avaliação do modelo real ainda dependem do dataset, código original, pesos e framework. Consulte o [guia](../docs/GUIA_DE_USO.md) e o [guia de componentes](../docs/plano-de-implementacao/COMPONENTES.md).

Os scripts podem ser chamados no terminal ou em automações; compartilham as funções Python usadas pelo projeto:

| Script | Finalidade |
| --- | --- |
| `experiment.py` | Criar ou consultar uma execução sintética. |
| `evaluate.py` | Avaliar um CSV de predições binárias já existentes e exportar tabela, métricas e gráficos. |
| `split_dataset.py` | Criar uma divisão por grupo a partir de um manifesto JSON, sem estratificação. |
| `download_dataset.py` | Baixar a versão fixada do dataset RLVS para `data/raw/`, com metadados de origem. |
| `research.py` | Preparar vídeos, treinar a reconstrução proposta e avaliar teste, robustez ou base externa no servidor. |

Execute `uv run python scripts/evaluate.py --help` ou `uv run python scripts/split_dataset.py --help` para ver os parâmetros. O primeiro exige as colunas `sample_id`, `true_label` e `predicted_label`; uma coluna `score` é opcional. O segundo espera `dataset_name`, `source`, `license` e `records` no JSON, com os campos de registro descritos no guia de componentes. Ambos exigem um caminho de saída novo e recusam sobrescrita.

O [guia de execução no servidor](../docs/plano-de-implementacao/EXECUCAO_NO_SERVIDOR.md) descreve os comandos `research.py prepare`, `train` e `evaluate`, os requisitos de GPU e os limites da reconstrução.
