# Memória e Contexto Estratégico

## Estratégia de Submissão (04/09/2026)
A proposta foi redigida deliberadamente com um escopo amplo, oferecendo múltiplas vias de expansão técnica (Vision Transformers, Edge AI, Detecção Não-Supervisionada, Dashboards de BI e IA Explicável/Grad-CAM). 
O objetivo dessa amplitude não é executar todas as frentes, mas sim **provocar o corpo editorial** a direcionar qual linha técnica tem maior alinhamento com o livro, evitando o risco de "dispersão metodológica".

## Lições do Feedback Editorial Anterior
Com base no feedback de um capítulo anterior aprovado pelo mesmo comitê:
1. **Profundidade > Amplitude:** O editor rejeita "dispersão metodológica" (fazer muitas coisas rasas). Quando o feedback retornar, devemos focar profundamente na via escolhida por eles.
2. **Posicionamento Conceitual:** O capítulo não pode parecer um artigo de informática médica/computação. A Visão Computacional é apenas o *caso empírico*. O tema central deve ser sempre a **Prontidão da Infraestrutura de Dados (Data Infrastructure Readiness)** e a Governança.
3. **Vocabulário Chave:** Expressões como "Algorithmic Readiness" e "Data Infrastructure" foram injetadas na proposta para alinhar a pesquisa à linguagem exata exigida pelos editores.

## Retorno Editorial (06/09/2026)

A aposta descrita acima funcionou. A proposta foi aprovada e os editores fizeram exatamente o que se esperava deles: apontaram a linha a seguir e mandaram cortar o resto. A aprovação é condicional, vale só para a fase de proposta, e está sujeita ao cumprimento de cinco exigências no capítulo completo.

### As cinco exigências

1. **Originalidade.** O comitê identificou o preprint arXiv:2502.18555 e listou nominalmente o que já está publicado: arquitetura CNN-BiLSTM-atenção, os três backbones, frames 100x100, configurações experimentais e os 96,5%. O capítulo tem de citar o trabalho anterior e demonstrar o que acrescenta. Parafrasear não conta como contribuição original e pode ser tratado como autoplágio.
2. **Contribuição de governança territorial.** É aqui que o capítulo se justifica no livro. Exigem um contexto territorial ou institucional específico, preferencialmente um município ou área urbana no Amazonas, respondendo sete perguntas: qual instituição opera, como o alerta entra no processo decisório, quem valida, qual critério justifica intervenção, como se tratam falsos positivos e negativos, como se documenta e audita, e como se mede valor público. A tecnologia tem de aparecer como apoio à decisão sob supervisão humana, nunca como despacho policial autônomo.
3. **Metodologia.** Mandaram reduzir escopo, citando nominalmente as cinco frentes da isca (ViT, Edge AI, detecção não supervisionada, Grad-CAM, dashboards de BI) mais fusão multimodal. Exigem separar o que foi feito do que é só proposto, e documentar dataset, amostra, distribuição de classes, anotação, split treino/validação/teste, prevenção de vazamento, hardware, parâmetros e reprodutibilidade. Métricas além da acurácia: precisão, recall, especificidade, F1 por classe, taxa de falso alerta, intervalos de confiança, latência de inferência e robustez sob variação de iluminação, câmera, aglomeração e oclusão. Validação externa ou cross-dataset é apontada como especialmente importante.
4. **Salvaguardas éticas e legais.** LGPD, avaliação de impacto à proteção de dados, proporcionalidade, minimização, retenção de imagens, controle de acesso, viés algorítmico, discriminação territorial, supervisão humana, accountability e mecanismos de revisão. Alerta explícito para não confundir detecção de evento observado com predição de comportamento criminoso futuro. O termo "predictive security", usado na proposta, deve ser revisto com cautela.
5. **Autoria.** Passamos de dois autores no trabalho anterior para sete na proposta. Exigem declaração no padrão CRediT com o papel de cada um, e contribuição intelectual substantiva de todos.

### Pendências mapeadas em 20/09/2026

**Dataset (bloqueante, aguardando o Erick).** Nenhum documento do projeto descreve a base usada. O artigo completo é a versão mais detalhada que existe e não traz seção de dataset. Faltam origem e licença dos vídeos, quantidade e distribuição de classes, critério e responsável pela anotação, split treino/validação/teste, hardware, épocas, logs e pesos. Há também a dúvida sobre se os 96,5% são de teste ou da última época de treino, já que o texto diz "accuracy obtained during training". O Erick é a única fonte, e o pedido é por arquivos, não por resposta de memória.

**Recorte territorial (em aberto, decisão não tomada).** Não há interlocutor institucional, nem contato em CIOPS, SSP-AM, guarda municipal ou prefeitura de Manaus. Avaliou-se o caminho documental, montar o arranjo institucional a partir de fonte pública (decretos de estrutura da SSP-AM e do CIOPS, legislação do SUSP, LGPD, planos municipais de videomonitoramento), já que o editor pede desenho institucional proposto e não estudo de caso empírico. Nesse caminho, o capítulo teria de declarar explicitamente que o arranjo é proposta normativa construída sobre documentos, sem validação com gestores, e adotar um recorte estreito em vez de Manaus inteira. Decisão adiada.

**Publicação anterior (fechada).** Só existe o preprint do arXiv, publicado por segurança. O IV CIPPCD não publicou anais. Não há segunda publicação a ser detectada pelo similarity screening.
