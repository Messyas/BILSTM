# Arquitetura do projeto

## Objetivo

Organizar experimentos reproduziveis de classificacao de videos do RLVS. Os notebooks explicam cada etapa, chamam uma API unica e apresentam parametros e resultados. O fluxo implementado usa uma reconstrucao proposta MobileNetV2 + BiLSTM + atencao; a equivalencia com o codigo e o checkpoint originais nao esta confirmada.

## Organizacao

```text
notebooks/                  interfaces das etapas reais
scripts/research.py         CLI para preparar, treinar e avaliar
scripts/download_dataset.py downloader do RLVS
src/bilstm_pipeline/        implementacao reutilizavel
configs/                    parametros TOML reproduziveis
data/raw/                   videos originais, fora do Git
data/processed/             manifestos e derivados, fora do Git
output/<run-id>/             parametros e resultados, fora do Git
docs/                       metodo, contexto e decisoes
```

| Caminho | Responsabilidade |
| --- | --- |
| `notebooks/` | Objetivo, configuracao, chamadas a API e apresentacao dos artefatos. |
| `scripts/` | Entradas finas para download e pipeline de pesquisa. |
| `src/bilstm_pipeline/` | Leitura de videos, validacao, divisao, treino, avaliacao, metricas e graficos. |
| `configs/` | Parametros cientificos e opcoes reproduziveis em TOML. |
| `data/` | Dataset original e artefatos derivados, fora do Git. |
| `output/` | Execucoes identificadas e seus artefatos, fora do Git. |

## Contrato dos notebooks

Os notebooks `01_prepare_dataset.ipynb`, `02_train_baseline.ipynb`, `03_evaluate_test.ipynb` e `04_evaluate_external.ipynb` cobrem as etapas disponiveis. Use ingles em nomes, identificadores, comandos e textos dos notebooks; comentarios e docstrings em codigo devem estar em ingles e portugues.

Cada notebook importa apenas a API de alto nivel do pacote. Nao configure ambiente ou graficos, nem coloque caminhos locais, processamento, metricas, treino ou imports de bibliotecas cientificas nas celulas. Parametros de experimento pertencem aos arquivos TOML em `configs/`. Os notebooks chamam a implementacao compartilhada e exibem artefatos associados ao ID da execucao.

## Execucao e artefatos

O fluxo de terminal esta no [guia do servidor](EXECUCAO_NO_SERVIDOR.md). `scripts/research.py` oferece `prepare`, `train` e `evaluate`; `scripts/download_dataset.py` baixa o RLVS. Cada treino cria `output/<run-id>/` com manifesto, configuracao efetiva, ambiente, historico e checkpoint. Cada avaliacao cria uma subpasta com metricas, previsoes, figuras e tempos. Execucoes nao devem sobrescrever artefatos existentes.

O processamento atual classifica um clipe offline. A avaliacao de robustez aplica perturbacoes configuradas aos videos; isso nao equivale a testar todas as condicoes de cameras reais. Resultados cientificos dependem de execucao com os dados e o protocolo aprovados.

## Figuras e manutencao

`plots.py` centraliza o estilo e exporta PDF e PNG via `matplotlib.rc_context`, preservando parametros globais. Dimensoes, formatos e fontes finais devem seguir as regras editoriais. Notebooks apenas exibem arquivos produzidos pelo pacote.

Antes de mudar codigo, leia o backlog, confira dependencias e preserve alteracoes locais. Atualize este guia quando mudar comandos ou responsabilidades; registre evidencias no [backlog](BACKLOG.md). Nao crie estrutura adicional sem necessidade concreta. Nao declare resultados inventados nem use o conjunto de teste para ajustar o modelo.
