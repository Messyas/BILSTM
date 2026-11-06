# Notebooks

Use notebooks como interface de cada experimento: explique a pergunta, apresente os parâmetros experimentais, chame a API de alto nível planejada e mostre tabelas e gráficos de `output/`. A API também será usada pelos comandos em `scripts/`.

Convenção de nomes: `NN_assunto.ipynb`, por exemplo `01_baseline.ipynb`, `02_metricas.ipynb`, `03_robustez.ipynb`. Cada notebook deve apontar para a configuração e o ID de execução usados. Apenas parâmetros científicos explícitos podem ser declarados nas células; a configuração efetiva deve ser registrada pelo pipeline. Não adicione configuração de ambiente, estilo de gráficos, treino, pré-processamento, métricas, nem imports diretos de Matplotlib, Seaborn, pandas ou frameworks de treino. Prefira uma única importação da API do projeto.

Para abrir: na raiz do projeto, rode `uv run --group notebooks jupyter lab` e selecione o kernel `Python (BILSTM Experimentos)`. O pipeline ainda será implementado após a chegada do código e dataset de origem; não há notebook de experimento executável por enquanto. Veja o [exemplo de interface e a arquitetura planejada](../docs/plano-de-implementacao/ARQUITETURA.md).
