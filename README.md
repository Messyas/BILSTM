# Experimentos de detecção de conflitos em vídeo

Este repositório organiza os experimentos computacionais de um capítulo sobre **prontidão da infraestrutura de dados e governança territorial**. O caso empírico é a detecção de conflitos humanos em vídeo com uma CNN + BiLSTM; o foco técnico atual é aprofundar a avaliação do modelo MobileNetV2 + BiLSTM, com atenção, e produzir resultados reproduzíveis que apoiem a discussão do capítulo.

Os notebooks são a interface de experimentação: apresentam objetivo, parâmetros experimentais, execução e gráficos. O pacote Python reutilizável é chamado pelos notebooks e por scripts de linha de comando. O dataset RLVS foi identificado pelo pesquisador; o código original, os pesos e os artefatos de execução ainda precisam ser recebidos, portanto os experimentos científicos não estão prontos para rodar.

## Começar

Instale o [uv](https://docs.astral.sh/uv/getting-started/installation/) e, na raiz do repositório, execute:

```powershell
uv sync --all-groups
uv run --group notebooks python -m ipykernel install --user --name bilstm-experimentos --display-name "Python (BILSTM Experimentos)"
uv run --group notebooks jupyter lab
```

No Jupyter, selecione o kernel **Python (BILSTM Experimentos)**. Os notebooks em `notebooks/` incluem uma visão geral e análises separadas de avaliação, robustez e divisão por grupos. Eles usam dados sintéticos e não executam o modelo real. Também é possível executar a demonstração pelo terminal:

```powershell
uv run python scripts/experiment.py run --config configs/synthetic_demo.toml
```

O comando informa um ID e grava os arquivos em `output/<id>/`. Veja o [tutorial de uso](docs/GUIA_DE_USO.md) para consultar uma execução anterior e conhecer as convenções do projeto.

### Baixar o dataset

O pesquisador identificou o [Real Life Violence Situations Dataset no Kaggle](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) como a base utilizada. Após configurar o acesso ao Kaggle conforme o [guia](docs/GUIA_DE_USO.md#2-baixar-o-dataset), rode na raiz:

```powershell
uv run --group data python scripts/download_dataset.py
```

O comando baixa a versão 1 em `data/raw/rlvs-v1/` e registra a origem e o SHA-256 do arquivo baixado. Os vídeos ficam fora do Git. A versão e as partições usadas no trabalho anterior ainda precisam ser confirmadas.

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

A origem do dataset foi informada pelo pesquisador; versão utilizada originalmente, licença e condições de uso, distribuição de classes, anotação, split e artefatos ainda precisam ser documentados. A acurácia de 96,5% mencionada em documentos de contexto não é tratada aqui como resultado de teste confirmado.

## Ambiente

O `pyproject.toml` descreve as dependências comuns de análise e notebook; `uv.lock` fixa as versões resolvidas para reproduzir esse ambiente. O framework de deep learning será adicionado quando o código e os pesos originais forem recebidos e a compatibilidade com o servidor/GPU for conhecida.
