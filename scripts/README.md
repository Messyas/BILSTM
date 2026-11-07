# Pipeline Python

Este diretório contém entradas finas de linha de comando. Elas interpretam argumentos e chamam a mesma API Python de alto nível usada pelos notebooks. A implementação compartilhada fica em `src/bilstm_pipeline/`; não duplique lógica de processamento ou gráficos nos scripts.

Etapas previstas pelo plano de experimentos: preparação e split dos dados por vídeo/cena; treino e validação; avaliação com métricas expandidas e intervalos de confiança; testes de robustez; e avaliação externa (*cross-dataset*). A implementação deve vir do código recebido e das decisões documentadas, sem duplicar a lógica em notebooks.

`experiment.py` executa e consulta uma demonstração com dados sintéticos. Os componentes reutilizáveis de métricas, divisão e robustez estão disponíveis; treino e avaliação do modelo real ainda dependem do dataset, código original, pesos e framework. Consulte o [guia](../docs/GUIA_DE_USO.md) e o [guia de componentes](../docs/plano-de-implementacao/COMPONENTES.md).
