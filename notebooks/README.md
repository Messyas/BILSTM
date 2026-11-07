# Notebooks

Use notebooks como interface de cada experimento: explique a pergunta, apresente os parâmetros experimentais, chame a API de alto nível e mostre tabelas e gráficos de `output/`. A API também é usada pelos comandos em `scripts/`.

Convenção de nomes em inglês: `NN_topic.ipynb`, por exemplo `01_baseline.ipynb`, `02_metrics.ipynb`, `03_robustness.ipynb`. Cada notebook deve apontar para a configuração e o ID de execução usados. Apenas parâmetros científicos explícitos podem ser declarados nas células; a configuração efetiva deve ser registrada pelo pipeline. Não adicione configuração de ambiente, estilo de gráficos, treino, pré-processamento, métricas, nem imports diretos de Matplotlib, Seaborn, pandas ou frameworks de treino. Prefira uma única importação da API do projeto. Texto dos notebooks é em inglês; comentários de código são em inglês e português.

Para abrir: na raiz do projeto, rode `uv run --group notebooks jupyter lab` e selecione o kernel `Python (BILSTM Experimentos)`. O notebook `00_infrastructure_demo.ipynb` funciona com dados sintéticos; o pipeline científico depende do código e dataset de origem. Veja a [arquitetura](../docs/plano-de-implementacao/ARQUITETURA.md).
