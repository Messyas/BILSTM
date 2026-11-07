# Avaliação crítica das fontes e do plano de experimentos

**Data:** 06/11/2026  
**Objetivo:** decidir se a proposta de Gustavo é uma base adequada para os experimentos do capítulo e quais condições precisam ser resolvidas antes da execução.

**Atualização em 07/11/2026:** o usuário identificou o [RLVS no Kaggle](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) como o dataset utilizado. A análise abaixo foi feita antes dessa informação; as pendências relativas à identidade da base estão parcialmente resolvidas, mas versão, amostras, divisão, licença e histórico de treinamento ainda precisam ser documentados. A versão 1 pode ser obtida pelo script descrito no [guia de uso](../GUIA_DE_USO.md#2-baixar-o-dataset).

## 1. Parecer para decisão

**A direção do plano faz sentido, com ajustes antes da execução.** Reavaliar o modelo existente com separação adequada dos dados, métricas mais completas, testes de robustez e avaliação externa responde às lacunas identificadas nos documentos internos. Essas etapas podem gerar evidência nova sobre o modelo sem exigir uma nova arquitetura. Porém, o plano ainda é um roteiro de trabalho, não um protocolo experimental executável: faltam a origem dos dados, as regras de avaliação, os artefatos do modelo e decisões sobre o que será medido.

Recomenda-se **aprovar a direção condicionada** à recuperação dos materiais originais e à definição prévia de um protocolo. Começar por inventário e reprodução do modelo; só então fixar a avaliação principal, os testes de robustez e uma base externa independente. O resultado desses experimentos poderá sustentar uma discussão de prontidão técnica; não demonstrará, por si só, prontidão institucional, impacto na segurança ou valor público.

**Limite da revisão editorial:** não foi encontrada em `docs/` a comunicação original dos editores. As cinco exigências são conhecidas aqui pelo [contexto interno](../Material-original/CONTEXTO.md) e pelo [relatório de Gustavo](<../Revisao-gustavo/Relatório do Capítulo IGI Global Resumo, Exigências e Plano de Ação.md>). Portanto, a correspondência exata com o pedido editorial e expressões como “mandaram cortar” devem ser confirmadas na mensagem original. Esta avaliação distingue o que está nas fontes de recomendações metodológicas próprias.

## 2. O que cada fonte efetivamente mostra

| Fonte | Evidência relevante | Limite para a decisão |
| --- | --- | --- |
| [Proposta final em inglês](<../Material-original/proposal/final_proposal.docx>) e [texto em português](<../Material-original/proposal/Texto IGI Palhares.docx>) | Propõem detecção de conflitos para apoio à gestão territorial. A versão final inclui CNN/BiLSTM/atenção, ViT, borda, anomalias, dashboards e Grad-CAM, além de indicadores de valor público. | São propostas amplas; a redação usa linguagem de predição, despacho policial e impacto que precisa ser revista à luz das exigências relatadas. Uma proposta não comprova realização ou impacto. |
| [Artigo completo, versão CIPPCD](<../Material-original/references/paper_full_v6_CIPPCD2025.pdf>), pp. 7–9 | Descreve entrada de **15 quadros RGB de 100 × 100**, saída binária, 12 configurações com três CNNs, com e sem atenção; a tabela 1 informa 96,50% para MobileNetV2 com atenção, `min_lr=0,00005` e lote 64. A tabela 2 já apresenta F1 por classe e a figura 6, uma matriz de confusão. | O texto fala em acurácia obtida durante o treinamento e não permite confirmar com segurança de que partição vêm os 96,50%, o F1 e a matriz. A identidade, a composição e o protocolo dos dados não estão suficientemente documentados. |
| [Outra versão do artigo](<../Material-original/references/Artigo_Aranoua_IFAM_Erick.pdf>), pp. 5–7 | Apresenta arquitetura, tabela de configurações e F1 por classe compatíveis com a descrição geral da versão acima. Lista quatro autores. | É uma versão local do texto; sua existência, por si, não determina seu status de publicação. |
| [Livro de resumos e artigos estendidos](<../Material-original/references/LIVRO_RESUMOS_CDPP_2025_compressed.pdf>), capa, p. 2, 46 e 410–419 do PDF | Identifica a **III Conferência Internacional de Políticas Públicas e Ciência de Dados**; contém, na p. 46, um resumo com dois autores e, nas pp. 410–419, o texto completo do artigo com quatro autores. | O livro local registra tanto o resumo quanto o texto completo. Verificar a referência bibliográfica, a divulgação do livro e a relação entre as versões; o exemplar não comprova a inexistência de outras publicações. |
| [Preprint arXiv:2502.18555](https://arxiv.org/abs/2502.18555) | A página pública lista Erick da Silva Farias e Eduardo Palhares Junior como autores. | Citar a versão publicada efetivamente usada e identificar seus autores conforme essa versão. |
| [Contexto](../Material-original/CONTEXTO.md), [tarefas](../Material-original/TODO.md) e [README original](../Material-original/README.md) | Registram aprovação da proposta, exigências, pendências e decisões em momentos diferentes. | São memória interna e podem estar desatualizados. O README original ainda anuncia dashboards e Edge AI, enquanto o plano posterior os retira do escopo prático. |
| [Plano de Gustavo](<../Revisao-gustavo/Plano de Revisão e Delegação de Simulações (Capítulo IGI Global).md>) e [relatório](<../Revisao-gustavo/Relatório do Capítulo IGI Global Resumo, Exigências e Plano de Ação.md>) | Priorizam reprodutibilidade, métricas, robustez e validação externa; reconhecem pendências de governança, ética e autoria. | Não trazem dados, código, pesos, rótulos, resultados novos nem protocolo completo. São propostas, não evidências de execução. |

### Inconsistências e cuidados de leitura

1. **Os 96,5% não são resultado de teste confirmado.** Na versão CIPPCD, a tabela 1 e o texto da p. 8 associam a métrica ao treinamento; a figura 6 da p. 9 mostra uma matriz com 400 observações e diagonal que soma 386, mas a figura isoladamente não prova que sejam vídeos independentes nem identifica a partição. Seus rótulos de coluna “True” e “False” também precisam ser conferidos no código. Recuperar predições, rótulos e logs antes de atribuir significado operacional ao valor.
2. **F1 por classe e matriz de confusão não são inéditos em todas as versões locais.** A contribuição nova deve ser uma avaliação reproduzível, com partição conhecida, métricas adicionais e incerteza, e não a simples existência dessas saídas. O artigo também apresenta comparações com/sem atenção e três backbones; repetir essas comparações sem novos dados ou pergunta claramente definida não constitui, automaticamente, novidade.
3. **Autoria varia por versão.** O preprint público e o resumo do livro citam dois autores; os dois PDFs locais completos citam quatro; a proposta do capítulo lista sete. O relatório de Gustavo se refere ao trabalho anterior com dois, enquanto o README interno menciona quatro. Não presumir erro ou equivalência entre documentos: conferir histórico, versão citada, contribuições e declaração CRediT com a equipe.
4. **Há resumo e artigo completo no livro local.** O livro da **III** conferência inclui o resumo na p. 46 e o texto completo nas pp. 410–419 do PDF. Isso corrige uma versão anterior desta avaliação que identificou apenas o resumo. A afirmação interna de que só existe o preprint e de que o trabalho foi apenas apresentado deve ser reconciliada com esse registro; ele não contradiz, por si só, uma afirmação específica sobre a **IV** conferência. Confirmar referências, versões e histórico com os autores antes de declarar as publicações anteriores e submeter à verificação de similaridade.
5. **Classificação binária.** A arquitetura descrita no artigo tem duas saídas (violência e não violência). A expressão “F-score multiclasse” da proposta precisa ser alinhada à tarefa real: para duas classes, informar F1 de cada classe e agregações definidas, sem afirmar que há múltiplos tipos de conflito identificados.

## 3. Análise do plano de Gustavo

| Bloco | Julgamento | Ajuste necessário antes de executar |
| --- | --- | --- |
| 1. Ambiente e reprodutibilidade | **Aceitar com condições.** Seeds, registro do ambiente e separação por vídeo/cena são essenciais. | Inventariar arquivos e versões; identificar unidade de independência, origem das cenas, pré-processamento, pesos e divisão histórica. Registrar a lista de IDs por partição e quaisquer decisões de determinismo. Seed isolada não garante reprodução entre dispositivos ou frameworks. |
| 2. Métricas e tempo | **Aceitar com alterações.** Precisão, recall, especificidade, F1, matriz, intervalos e tempo de inferência são pertinentes. | Definir classe positiva, unidade da predição e de análise, limiar, agregações e denominadores. Fixar o limiar na validação, preservar o teste e calcular intervalos no nível de vídeo/cena independente. Medir latência completa por sequência e, quando apropriado, por quadro. |
| 3. Robustez | **Aceitar em escopo controlado.** Perturbações podem revelar sensibilidade sob mudanças específicas. | Pré-registrar duas ou três condições e níveis justificados, comparando os mesmos vídeos com a versão limpa. Uma máscara não reproduz automaticamente uma aglomeração; não concluir robustez a câmeras, densidade ou condições reais em geral a partir de uma única transformação sintética. |
| 4. Validação externa | **Aceitar como prioridade condicionada à independência.** Avaliar dados de outra origem pode mostrar transferência para outro domínio. | Escolher a base depois de identificar a base de treino. Verificar licença, sobreposição de vídeos/cenas, classes, anotações e pré-processamento. Congelar modelo e limiar antes do teste externo e explicar as diferenças entre domínios. |
| 5. Prazo e delegação | **Usar apenas como estimativa inicial.** Os 8–14 dias úteis do relatório correspondem à soma sequencial dos intervalos propostos, após a entrega dos materiais. | Reestimar após inventário e execução piloto, considerando acesso a GPU, treinamento, obtenção e revisão de rótulos, correções e disponibilidade da equipe. Blocos 3 e 4 podem ser paralelos depois de modelo e protocolo congelados, se houver recursos e pessoas; não presumir paralelismo num único servidor. |

O relatório identifica a governança, a ética e a autoria como pendências, o que é correto. Entretanto, chamar o dataset de “único bloqueio imediato” simplifica demais: também podem faltar código executável, pesos, metadados, licença, ambiente, capacidade de processamento e protocolo. A escrita territorial e o levantamento de fontes podem começar em paralelo à espera dos dados.

## 4. Protocolo recomendado para decidir o que experimentar

### Etapa 0 — Inventário e perguntas bloqueantes

Solicitar ao responsável pelo trabalho anterior, preferencialmente como arquivos verificáveis:

- Código e histórico/versão exata; dependências, framework, scripts de pré-processamento e de inferência; arquitetura e pesos/checkpoints identificados.
- Dataset original, lista de vídeos, fontes, permissões/licenças, quantidade por classe, rótulos e critérios de anotação; IDs de vídeo, cena, origem e possíveis duplicatas ou recortes do mesmo original.
- Partições originais, sementes, definição de sequência, taxa de amostragem de quadros, resolução e normalização; parâmetros, épocas, logs, hardware e predições usadas nas tabelas e figuras.
- Como foram escolhidos o melhor modelo e o limiar, em que conjunto foram calculados os 96,5%, F1 e matriz, e o que representam as 400 observações da figura.

Documentar também como o modelo foi pré-treinado e se houve uso anterior da candidata a base externa, inclusive em seleção e ajuste. Sem esses dados, a independência da validação não pode ser afirmada.

### Etapa 1 — Reproduzir e congelar uma linha de base

1. Conferir se o checkpoint realmente implementa a arquitetura e o processamento descritos; validar que produz saídas para entradas conhecidas e registrar diferenças. Usar pesos e parâmetros originais se disponíveis.
2. Separar conjuntos por grupos independentes no nível mais alto identificável: vídeo de origem e, quando compartilhados, cena/câmera/evento ou fonte que possa gerar duplicatas. Todas as janelas derivadas de um grupo ficam na mesma partição. Informar o número de grupos, vídeos e janelas por classe em cada divisão.
3. Escolher 70/15/15 ou 80/10/10 **somente depois** de conhecer tamanho e distribuição de grupos. Caso haja poucos grupos, planejar uma alternativa como validação cruzada agrupada, preservando uma avaliação final independente quando viável; explicitar o limite se isso não for possível.
4. Selecionar configurações e limiar apenas com treino/validação; congelar pré-processamento, modelo e limiar antes de consultar o teste. Preservar listas de IDs e configuração para reproduzir a execução.

**Se houver pesos, mas não houver informação suficiente sobre seus dados de treino**, o teste com a base original pode conter exemplos já vistos; rotular a avaliação como exploratória, sem alegar teste independente. **Se houver código e dados, mas não pesos**, retreinar sob protocolo documentado e chamar o resultado de nova reprodução, não do checkpoint original. **Se código ou dados forem insuficientes**, registrar a impossibilidade de reproduzir e não apresentar resultados simulados como resultados do artigo.

### Etapa 2 — Avaliação principal

Definir antes da execução se a unidade de interesse é a **janela de 15 quadros**, o **vídeo**, um **evento** ou um **alerta**. Se o modelo gerar várias janelas por vídeo, fixar previamente como agregá-las e com que rótulos compará-las. Uma janela, um vídeo e um alerta operacional não são equivalentes.

Para a classe positiva “violência/conflito”, registrar matriz com rótulos explícitos, suporte por classe, precisão, recall/sensibilidade, especificidade, F1 por classe e médias macro e ponderada. Informar também acurácia como referência, sem usá-la isoladamente. Para FPR = FP/(FP+VN), especificar o que é um negativo e a unidade de contagem; **FPR não é automaticamente falsos alertas por hora**. Só calcular alertas por hora/câmera se houver vídeos contínuos, duração, regras de agregação, supressão/repetição e anotações adequadas. Reportar denominadores e casos em que uma métrica não é estimável.

Para IC de 95%, aplicar bootstrap nos **grupos independentes** (vídeo ou grupo de origem mais amplo), mantendo juntas suas janelas; registrar número de reamostragens, seed, método e tratamento de amostras sem uma classe. Para comparações limpo versus perturbado, usar reamostragem pareada dos mesmos grupos. Esse intervalo descreve a incerteza associada à amostra de avaliação com o modelo fixo; ele não captura sozinho a variabilidade do treinamento. Se houver tempo e capacidade, repetir treinos com seeds diferentes e relatar separadamente essa variação.

**Latência:** o modelo descrito usa uma sequência bidirecional de 15 quadros, portanto necessita da janela completa para classificá-la. Medir separadamente aquisição/formação da janela, pré-processamento, inferência, pós-processamento e latência ponta a ponta; registrar taxa de amostragem, passo entre janelas, batch, hardware, framework, aquecimento e sincronização quando houver GPU. Reportar distribuição (por exemplo, mediana e percentil 95), unidade e throughput sob carga definida. `ms/frame` derivado de uma sequência não demonstra, sozinho, latência de detecção nem viabilidade em tempo real. Uma BiLSTM pode operar em janelas acumuladas, mas depende dos quadros futuros **dentro da janela**; o atraso deve ser medido no protocolo escolhido.

### Etapa 3 — Robustez: pequena matriz de experimentos

Usar o mesmo conjunto de teste, as mesmas etiquetas e o modelo congelado para cada condição. Aplicar transformações de maneira temporalmente coerente dentro de cada vídeo, registrar código, seed e intensidade, e inspecionar uma amostra para verificar se o rótulo continua válido. Reportar métricas limpas e perturbadas, diferenças pareadas e incerteza; descrever condições em que a alteração impede uma interpretação segura.

| Prioridade proposta | Condição | O que permite concluir |
| --- | --- | --- |
| 1 | Brilho/contraste em dois níveis pré-definidos | Sensibilidade às transformações implementadas, não desempenho comprovado em todas as cenas noturnas. |
| 2 | Compressão ou ruído em dois níveis pré-definidos | Sensibilidade à degradação simulada escolhida, não a todas as câmeras de vigilância. |
| 3 | Oclusão parcial, com posição e proporção registradas | Sensibilidade à máscara aplicada. Não é equivalente a aumento real de pessoas na cena. |

Para aglomeração, preferir uma avaliação separada com rótulos ou metadados de densidade e amostras suficientes. Sem eles, declarar essa dimensão como não avaliada. Escolher as condições e intensidades antes de olhar os resultados finais; não selecionar apenas as que produzem maior queda.

### Etapa 4 — Validação externa

Selecionar uma base externa que não tenha contribuído para treino, ajuste, seleção de hiperparâmetros ou limiar, inclusive por vídeos duplicados ou reedições. Verificar as licenças, condições de acesso e rótulos. Mapear classes previamente: escolher qual evento conta como positivo e quais exemplos podem ser negativos, excluindo ou analisando à parte categorias ambíguas. Documentar regras de amostragem temporal, resolução, normalização e agregação por vídeo; manter as mesmas entradas esperadas pelo modelo. Só chamar a avaliação de **zero-shot entre bases** se não houver treinamento nem ajuste com dados ou rótulos da base alvo; qualquer adaptação deve ser relatada separadamente.

- **RLVS:** identificada posteriormente pelo usuário como base do trabalho original. Uma avaliação nessa base pode servir para reprodução ou teste interno com divisão independente documentada; ela não é, por si, validação externa entre datasets para um modelo treinado nela.
- **Hockey Fight:** candidato para estudar mudança de domínio esportivo; resultados nesse domínio não representam diretamente videomonitoramento urbano.
- **UCF-Crime:** candidato de vigilância mais próximo de algumas condições de interesse, porém a [descrição dos autores](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/) inclui vídeos longos e 13 tipos de anomalia, muitos não equivalentes a conflito entre pessoas. Rótulos de anomalia em nível de vídeo não podem ser transferidos indiscriminadamente a todas as janelas. Definir categorias, recortes e anotações temporais disponíveis; não rotular toda anomalia como violência ou toda janela de um vídeo positivo como positiva.

Apresentar resultados por base e escopo, tamanho e distribuição das amostras, limiar, IC e limitações. Uma base externa favorável é evidência de desempenho naquele domínio, não prova de generalização para qualquer território.

## 5. Ligação com governança e as exigências relatadas

| Exigência relatada | Contribuição possível dos experimentos | Trabalho adicional necessário |
| --- | --- | --- |
| Originalidade | Novas medições documentadas de erro, robustez e transferência, se forem realizadas. | Citar trabalhos anteriores, explicar o que já existia e declarar exatamente o que foi acrescentado. Confirmar histórico de autoria e resumo da conferência. |
| Governança territorial | Mostrar quais limitações técnicas afetariam um possível fluxo de alertas. | Definir território e instituição, responsáveis, validação humana, critérios de intervenção, tratamento de falsos alertas e falhas, auditoria e indicadores. Um desenho baseado em documentos públicos deve ser identificado como proposta sem validação institucional, quando for o caso. |
| Metodologia | Protocolo, dados, split, métricas, robustez, incerteza, tempo e validação externa. | Completar e executar o protocolo, fornecer evidências e separar resultados de propostas. |
| Ética e proteção de dados | Métricas e análise de erros podem informar riscos. | Avaliar base legal e salvaguardas aplicáveis, proporcionalidade, retenção, acesso, vieses, contestação e supervisão humana com apoio especializado. Métricas de laboratório não demonstram conformidade com a LGPD. |
| Autoria | Logs e contribuições ajudam a documentar o trabalho. | Confirmar contribuições intelectuais de cada autor e preparar CRediT conforme a versão efetivamente submetida. |

Indicadores como precisão ou velocidade não medem, por si, redução de violência, economia pública, prontidão institucional ou confiança social. Esses efeitos exigem definição de indicadores, contexto, fontes e avaliação apropriada. O sistema deve ser apresentado como apoio à análise humana de eventos observados, sem inferir intenção criminal futura ou despacho automático.

## 6. Ordem prática e critérios de parada

1. **Agora:** obter a mensagem editorial original e os materiais da etapa 0; confirmar autoria, versões, publicação do resumo e histórico das bases. Em paralelo, decidir recorte territorial e responsáveis pelas frentes de governança, ética e redação.
2. **Após inventário:** escolher o cenário viável (pesos originais, retreinamento documentado ou impossibilidade de reprodução); fechar protocolo de separação, métricas, limiar, unidade de análise e registro de execução.
3. **Executar linha de base:** comprovar que pipeline e dados correspondem ao protocolo e gerar relatório com métricas e incerteza. A ausência de informação suficiente sobre independência impede apresentar o resultado como teste independente.
4. **Executar robustez e validação externa:** congelar modelo e protocolo; manter as análises pareadas e documentar condições, rótulos e origem. Parar a comparação externa se não for possível excluir contaminação ou definir rótulos compatíveis; relatar a limitação.
5. **Integrar ao capítulo:** separar o que foi executado do proposto, comparar com o trabalho anterior e discutir limites técnicos no fluxo territorial. Não adiar governança e ética até o fim dos experimentos.

**Decisões que a equipe precisa confirmar:** qual versão/checkpoint será a referência; de onde vem o dataset original e se contém uma das bases candidatas; qual unidade e agrupamento dos dados; qual base externa realmente independente e licenciada; quais perturbações cabem no prazo; quem define e valida o arranjo territorial; e qual histórico de publicação deve constar na submissão.

## 7. Fontes externas consultadas

- [Página do preprint arXiv:2502.18555](https://arxiv.org/abs/2502.18555): identificação e autores da versão pública.
- [Página dos autores do UCF-Crime](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/): escopo, categorias, duração dos vídeos e anotações disponibilizadas.

As demais observações se apoiam nos arquivos locais apontados na seção 2 ou são recomendações metodológicas desta avaliação. Não houve execução de experimentos, verificação de código original ou validação institucional nesta análise.
