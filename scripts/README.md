# Pipeline Python

Este diretório conterá os comandos reutilizáveis do pipeline. Cada etapa deve aceitar parâmetros explícitos por linha de comando ou arquivo em `configs/`, registrar a configuração efetiva e escrever resultados em `output/<id-da-execucao>/`.

Etapas previstas pelo plano de experimentos: preparação e split dos dados por vídeo/cena; treino e validação; avaliação com métricas expandidas e intervalos de confiança; testes de robustez; e avaliação externa (*cross-dataset*). A implementação deve vir do código recebido e das decisões documentadas, sem duplicar a lógica em notebooks.

O pipeline ainda não foi implementado porque dataset, código original, pesos e framework precisam ser confirmados. Consulte `docs/GUIA_DE_USO.md` para o contrato de execução e convenções de pastas.
