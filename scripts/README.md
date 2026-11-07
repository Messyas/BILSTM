# Pipeline Python

Este diretório contém entradas finas de linha de comando. Elas interpretam argumentos e chamam a mesma API Python de alto nível usada pelos notebooks. A implementação compartilhada fica em `src/bilstm_pipeline/`; não duplique lógica de processamento ou gráficos nos scripts.

Etapas previstas pelo plano de experimentos: preparação e split dos dados por vídeo/cena; treino e validação; avaliação com métricas expandidas e intervalos de confiança; testes de robustez; e avaliação externa (*cross-dataset*). A implementação deve vir do código recebido e das decisões documentadas, sem duplicar a lógica em notebooks.

`experimento.py` já executa e consulta uma demonstração com dados sintéticos. As etapas científicas ainda não foram implementadas: dataset, código original, pesos e framework precisam ser confirmados. Consulte o [guia](../docs/GUIA_DE_USO.md) para os comandos atuais e a [arquitetura](../docs/plano-de-implementacao/ARQUITETURA.md) para os contratos.
