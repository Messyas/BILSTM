# Notebooks

Use notebooks como interface de cada experimento: explique a pergunta, mostre os parâmetros, invoque os comandos em `scripts/` e apresente tabelas e gráficos gerados a partir de `output/`.

Convenção de nomes: `NN_assunto.ipynb`, por exemplo `01_baseline.ipynb`, `02_metricas.ipynb`, `03_robustez.ipynb`. Cada notebook deve apontar para os arquivos de configuração e resultados usados. Não implemente treino, pré-processamento ou métricas completas dentro das células.

Para abrir: na raiz do projeto, rode `uv run --group notebooks jupyter lab` e selecione o kernel `Python (BILSTM Experimentos)`. O pipeline ainda será implementado após a chegada do código e dataset de origem; não há notebook de experimento executável por enquanto.
