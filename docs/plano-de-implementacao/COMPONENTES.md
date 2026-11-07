# Componentes do pipeline

O pacote `src/bilstm_pipeline/` contem funcoes reutilizaveis chamadas pelos notebooks e por `scripts/research.py`. Verificacoes unitarias podem usar casos de entrada controlados para conferir o software; suas saidas nao sao resultados de pesquisa.

## Dados e particoes

`research_data.py` inventaria videos, le classes pelos diretorios `Violence` e `NonViolence`, calcula hashes, cria manifesto e decodifica sequencias. O padrao agrupa arquivos com SHA-256 identico. Para manter cenas ou origens juntas, `research.py prepare --groups grupos.csv` recebe um mapeamento completo e revisado `sample_id,group_id`. Hashes nao identificam automaticamente reedicoes ou cenas relacionadas.

`dataset.py` valida registros, manifestos e ausencia de intersecao entre grupos, videos e hashes conhecidos nas particoes. `research_config.py` valida parametros TOML e registra configuracao efetiva por execucao.

## Treino e avaliacao

`research_model.py` implementa a reconstrucao MobileNetV2 + BiLSTM com atencao. `research_training.py` prepara lotes, treina nas particoes de treino e validacao e seleciona checkpoint pela perda de validacao. `research_evaluation.py` avalia o conjunto de teste ou uma base externa, registra metricas e tempos e pode aplicar perturbacoes configuradas.

`metrics.py` calcula metricas binarias e intervalos por bootstrap agrupado, inclusive comparacoes pareadas. `results.py` valida e grava previsoes e metricas. `robustness.py` oferece transformacoes de brilho, contraste, ruido, oclusao e compressao JPEG. E preciso revisar efeitos e validade dos rotulos para cada protocolo real.

## Graficos e notebooks

`plots.py` produz matrizes de confusao, metricas binarias, comparacoes de robustez e historico de treino em PDF e PNG. Usa `matplotlib.rc_context` para preservar o estilo global. Notebooks chamam funcoes de `notebook.py` e nao importam diretamente bibliotecas de processamento ou desenho.

Os notebooks sao `01_prepare_dataset.ipynb`, `02_train_baseline.ipynb`, `03_evaluate_test.ipynb` e `04_evaluate_external.ipynb`. O guia do servidor documenta parametros, comandos e limitacoes.
