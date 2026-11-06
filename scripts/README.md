# Pipeline Python

Este diretório conterá entradas finas de linha de comando. Elas interpretarão argumentos e chamarão a mesma API Python de alto nível usada pelos notebooks. A implementação compartilhada ficará no pacote planejado `src/bilstm_pipeline/`; não duplique lógica de processamento ou gráficos nos scripts. Cada etapa deve aceitar parâmetros explícitos ou arquivo em `configs/`, registrar a configuração efetiva e escrever resultados em `output/<id-da-execucao>/`.

Etapas previstas pelo plano de experimentos: preparação e split dos dados por vídeo/cena; treino e validação; avaliação com métricas expandidas e intervalos de confiança; testes de robustez; e avaliação externa (*cross-dataset*). A implementação deve vir do código recebido e das decisões documentadas, sem duplicar a lógica em notebooks.

O pipeline ainda não foi implementado porque dataset, código original, pesos e framework precisam ser confirmados. Consulte o [guia](../docs/GUIA_DE_USO.md) e a [arquitetura planejada](../docs/plano-de-implementacao/ARQUITETURA.md) para os contratos e as convenções.
