# Experimentos de detecção de conflitos em vídeo

Este repositório organiza os experimentos computacionais de um capítulo sobre **prontidão da infraestrutura de dados e governança territorial**. O caso empírico é a detecção de conflitos humanos em vídeo com uma CNN + BiLSTM; o foco técnico atual é aprofundar a avaliação do modelo MobileNetV2 + BiLSTM, com atenção, e produzir resultados reproduzíveis que apoiem a discussão do capítulo.

Os notebooks são a interface de experimentação: apresentam objetivo, parâmetros experimentais, execução e gráficos. O pacote Python reutilizável é chamado pelos notebooks e por scripts de linha de comando. O dataset RLVS e os notebooks originais de extração (`FEATURE_EXTRACTION.ipynb`) e treinamento (`parametro_alterado_definitivo_densenet121_att(1).ipynb`) foram enviados por Erick Farias; a arquitetura (escolha entre `MobileNetV2`, `DenseNet121` e `InceptionV3`, BiLSTM de 32 unidades, atenção por soma ponderada e 4 camadas densas) e os hiperparâmetros (`SGD`, `EarlyStopping` e `ReduceLROnPlateau`) estão integrados ao pipeline em PyTorch, enquanto os arquivos de pesos `.keras` originais não foram anexados.

## Começar

Instale o [uv](https://docs.astral.sh/uv/getting-started/installation/) e, na raiz do repositório, execute:

```powershell
uv sync --all-groups
uv run --group notebooks python -m ipykernel install --user --name bilstm-experimentos --display-name "Python (BILSTM Experimentos)"
uv run --group notebooks jupyter lab
```

No Jupyter, selecione o kernel **Python (BILSTM Experimentos)**. Os notebooks `01` a `04` cobrem preparo, treino, avaliacao no teste reservado e avaliacao externa. O treino e as avaliacoes devem rodar no servidor configurado. Consulte o [guia de uso](docs/GUIA_DE_USO.md) para os comandos do fluxo real.

### Baixar o dataset

O pesquisador identificou o [Real Life Violence Situations Dataset no Kaggle](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) como a base utilizada. Após configurar o acesso ao Kaggle conforme o [guia](docs/GUIA_DE_USO.md#2-baixar-o-dataset), rode na raiz:

```powershell
uv run --group data python scripts/download_dataset.py
```

O comando baixa a versão 1 em `data/raw/rlvs-v1/` e registra a origem e o SHA-256 do arquivo baixado. Os vídeos ficam fora do Git. A versão e as partições usadas no trabalho anterior ainda precisam ser confirmadas.

### Executar no servidor

O [guia de execução no servidor](docs/plano-de-implementacao/EXECUCAO_NO_SERVIDOR.md) detalha preparação do RLVS, configuração, treino, avaliação, robustez e validação externa. O modelo PyTorch implementa a arquitetura e os hiperparâmetros dos notebooks originais (CNN selecionável entre MobileNetV2, DenseNet121 e InceptionV3 + BiLSTM + atenção opcional); como o código original foi executado em TensorFlow/Keras (TPU) e os arquivos `.keras` treinados não foram anexados, um novo treinamento deve ser identificado como adaptação/reprodução em PyTorch, e não como o checkpoint original. Confira a GPU e o PyTorch do servidor antes de executar.

## Mapa do projeto

| Caminho | O que deve conter |
| --- | --- |
| `notebooks/` | Interface dos experimentos: pergunta, parâmetros experimentais, chamadas ao pipeline, tabelas e gráficos. |
| `scripts/` | Entradas finas de linha de comando que chamam a mesma API dos notebooks. |
| `src/bilstm_pipeline/` | Pacote Python para parâmetros, divisão por grupos, métricas, gráficos, robustez e interface de modelo. |
| `configs/` | Parâmetros versionados por experimento ou protocolo, usados para gerar a configuração efetiva de cada execução. |
| `data/raw/` | Dados originais, mantidos fora do Git. Não editar os arquivos de origem. |
| `data/processed/` | Dados derivados e divisões de treino, validação e teste, também fora do Git. |
| `output/` | Resultados gerados, logs, checkpoints e figuras; por padrão, fora do Git. |
| `docs/` | Contexto da pesquisa, decisões, documentação e tutoriais. |

Consulte o [guia de uso](docs/GUIA_DE_USO.md) antes de criar arquivos ou pastas. A [arquitetura](docs/plano-de-implementacao/ARQUITETURA.md) descreve o fluxo, o [guia de componentes](docs/plano-de-implementacao/COMPONENTES.md) explica as funções já disponíveis e o [backlog](docs/plano-de-implementacao/BACKLOG.md) é a fonte de andamento. Agentes devem seguir também [AGENTS.md](AGENTS.md).

## Escopo científico atual

O plano de revisão prioriza métricas por classe e agregadas, matriz de confusão, taxa de falso alerta, intervalos de confiança, latência e throughput; testes de robustez; e validação externa (*cross-dataset*). A documentação também registra as exigências editoriais sobre originalidade, governança territorial, metodologia, salvaguardas éticas e autoria. O capítulo deve separar resultados efetivamente medidos de propostas futuras e tratar o sistema como apoio à decisão sob supervisão humana.

A origem do dataset e os notebooks originais de pré-processamento e treino foram recebidos: nos experimentos anteriores, os 2000 vídeos do RLVS foram divididos em 64% treino (1280), 16% validação (320) e 20% teste (400 vídeos, onde foram calculadas as acurácias de 96,50% e 95,50%, o F1 e a matriz de confusão), porém com divisão aleatória simples sem agrupamento por cena e com aumento de dados (`imgaug`) aplicado durante a extração antes da divisão. Licença e condições de uso, agrupamento por cenas e pesos `.keras` ainda precisam ser revisados.

## Ambiente

O `pyproject.toml` descreve as dependências de análise e os grupos `notebooks`, `data` e `training`; `uv.lock` registra as versões resolvidas. O grupo `training` usa PyTorch, TorchVision e OpenCV. A compatibilidade da instalação de PyTorch com a GPU deve ser verificada no servidor.
