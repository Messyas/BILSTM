# Relatório do Capítulo IGI Global: Resumo, Exigências e Plano de Ação

Nov 4, 2026 · @Gustavo

## 1. Resumo da proposta

O capítulo propõe um framework de deep learning para detectar conflitos humanos em vídeo em tempo real, como apoio à decisão de governos municipais sob supervisão humana. Foi aprovado condicionalmente pelo comitê editorial em 06/09/2026, só para a fase de proposta.

| Item | Descrição |
| --- | --- |
| Título | AI-Powered Urban Safety: Real-Time Human Conflict Detection and Decision-Support Systems for Smart Territories |
| Livro | AI-Driven Territorial Governance for Effective Public Decision-Making |
| Autores (7) | Eduardo Palhares Júnior, Erick da Silva Farias, Marcelo Chamy Machado, Alyson de Jesus dos Santos, Alexandre Lopes Martiniano, Wenndisson da Silva Souza, Eduardo Corrêa dos Santos (IFAM e FUCAPI) |
| Problema | O videomonitoramento manual não escala e sofre com fadiga visual, o que atrasa a resposta a crises de segurança pública |
| Modelo base | CNN + BiLSTM + atenção, com MobileNetV2 como melhor backbone e acurácia acima de 96% (96,5% segundo o comitê) |
| Método | Fase quantitativa: sequências de quadros 100x100, avaliadas por acurácia, F-Score multiclasse e matriz de confusão. Fase qualitativa: Grad-CAM e indicadores de valor público |
| Eixo conceitual | Prontidão algorítmica e da infraestrutura de dados municipais. A visão computacional é apenas o caso empírico |

A proposta foi escrita de forma ampla de propósito (ViT, Edge AI, detecção não supervisionada, Grad-CAM, dashboards de BI). A ideia era levar o comitê a escolher a linha técnica e evitar a dispersão metodológica. Deu certo: os editores mandaram cortar o resto e aprofundar uma via só.

## 2. As cinco exigências do comitê

A aprovação vale só para a proposta e depende do cumprimento destas cinco exigências no capítulo completo. O plano de delegação atual cobre a número 3 e parte da 1. As exigências 2, 4 e 5 ainda não têm plano nem responsável.

| # | Exigência | O que o comitê pede | Situação no plano |
| --- | --- | --- | --- |
| 1 | Originalidade | Citar o preprint arXiv:2502.18555 e mostrar o que o capítulo acrescenta. Já está publicado: arquitetura CNN-BiLSTM-atenção, três backbones, frames 100x100, configurações e os 96,5%. Paráfrase pode ser tratada como autoplágio | Parcial: os Blocos 2 a 4 geram os resultados novos, mas falta definir o texto de posicionamento frente ao preprint |
| 2 | Governança territorial | Contexto institucional específico, de preferência um município ou área urbana no Amazonas, respondendo sete perguntas: quem opera, como o alerta entra na decisão, quem valida, qual critério justifica intervenção, como tratar falsos positivos e negativos, como auditar e como medir valor público. A IA é apoio sob supervisão humana, nunca despacho autônomo | Sem plano. Decisão sobre o recorte territorial em aberto |
| 3 | Metodologia | Reduzir escopo, separar o feito do proposto e documentar dataset, anotação, split, prevenção de vazamento, hardware e reprodutibilidade. Métricas além da acurácia, robustez e validação cross-dataset | Coberta pelos Blocos 1 a 4. Depende do dataset do Erick |
| 4 | Salvaguardas éticas e legais | LGPD, avaliação de impacto, proporcionalidade, minimização, retenção de imagens, controle de acesso, viés, supervisão humana e revisão. Não confundir detecção de evento com predição de crime futuro. Rever o termo "predictive security" | Sem plano |
| 5 | Autoria | Declaração CRediT com o papel de cada um dos sete autores e contribuição intelectual substantiva de todos (o trabalho anterior tinha dois autores) | Sem plano |

## 3. Escopo: o que fica e o que sai

O foco é único: aprofundar o modelo base MobileNetV2 + BiLSTM, com rigor metodológico, e usar o caso como evidência empírica de prontidão da infraestrutura de dados e governança.

| Fica | Cortado do escopo prático |
| --- | --- |
| Modelo MobileNetV2 + BiLSTM com atenção | Vision Transformers |
| Métricas expandidas e intervalos de confiança | Edge AI |
| Testes de robustez | Detecção não supervisionada |
| Validação cross-dataset | Grad-CAM |
| Governança, ética e LGPD (texto do capítulo) | Dashboards de BI e fusão multimodal |

O que for cortado pode aparecer no capítulo como trabalho futuro, declarado como proposto e não realizado.

## 4. Plano de execução das simulações

Toda a execução técnica fica com Wenndisson, que tem acesso ao servidor. Somando os prazos em sequência, a fase leva de 8 a 14 dias úteis depois que o Erick enviar código e dados. Os Blocos 3 e 4 são independentes entre si e podem rodar em paralelo depois do Bloco 2.

| Etapa | Responsável | Tarefas principais | Prazo |
| --- | --- | --- | --- |
| Alinhamento e dataset | Erick / liderança | Enviar código, pesos, logs e base original de dados | Dependência imediata |
| Bloco 1: ambiente e reprodutibilidade | Wenndisson | Pipeline de treino, validação e teste. Seeds fixas. Log automático de CUDA, framework, learning rate, batch size e épocas. Split 70/15/15 ou 80/10/10 por vídeo/cena, contra vazamento de dados | 1 a 2 dias úteis |
| Bloco 2: métricas expandidas | Wenndisson | Precisão, recall, especificidade e F1 (por classe e ponderado). Matriz de confusão, taxa de falso alerta e IC 95% por bootstrapping. Latência (ms/frame) e throughput (FPS) | 1 a 3 dias úteis |
| Bloco 3: robustez | Wenndisson | Variação de brilho e contraste. Oclusão parcial e alta densidade de pessoas. Compressão de vídeo e ruído de CFTV | 3 a 5 dias úteis |
| Bloco 4: validação externa | Wenndisson | Avaliação zero-shot em dataset público (RLVS, UCF Crime ou Hockey Fight) em 100x100, com tabela comparativa de generalização | 3 a 4 dias úteis |
| Integração e redação | Equipe de redação | Escrever as seções e gerar tabelas e gráficos | Paralelo ou sequencial |

O Bloco 1 vem primeiro porque o split por vídeo/cena e as seeds fixas valem para todos os resultados seguintes.

## 5. Pendências e riscos

O dataset é o único bloqueio imediato: sem ele, nenhum bloco do plano começa. O recorte territorial é a decisão em aberto mais importante, porque é onde o capítulo se justifica no livro.

| Pendência ou risco | Situação (20/09/2026) | Ação |
| --- | --- | --- |
| Dataset (bloqueante) | Nenhum documento descreve a base. Faltam origem e licença dos vídeos, quantidade e distribuição de classes, critério e responsável pela anotação, split, hardware, épocas, logs e pesos | Pedir ao Erick os arquivos, não respostas de memória |
| Os 96,5% são de teste? | O texto diz "accuracy obtained during training". Pode ser a última época de treino e não o teste | Confirmar com os logs. O Bloco 2 refaz a medição em teste |
| Recorte territorial | Sem interlocutor em CIOPS, SSP-AM, guarda municipal ou prefeitura de Manaus. Decisão adiada | Decidir entre arranjo documental ou contato institucional (ver abaixo) |
| Exigências 2, 4 e 5 | Não estão no plano de delegação | Definir responsáveis e prazos |
| Autoplágio | O preprint arXiv já tem a arquitetura, os backbones e os 96,5%. Não há segunda publicação (o IV CIPPCD não publicou anais) | Citar o preprint e destacar só o que é novo |
| Termo "predictive security" | O comitê pediu cautela: detecção de evento observado não é predição de comportamento futuro | Revisar o termo em todo o texto |

**Caminho documental para o recorte territorial.** Montar o arranjo institucional a partir de fontes públicas: decretos de estrutura da SSP-AM e do CIOPS, legislação do SUSP, LGPD e planos municipais de videomonitoramento. O editor pede um desenho institucional proposto, não um estudo de caso empírico. Nesse caminho, o capítulo precisa declarar que o arranjo é uma proposta normativa baseada em documentos, sem validação com gestores, e adotar um recorte estreito em vez de Manaus inteira.

## 6. Próximos passos

- [ ] Erick envia código, pesos, logs e a base original, com origem, licença, classes, anotação e split
- [ ] Confirmar se os 96,5% são de teste ou de treino
- [ ] Wenndisson executa o Bloco 1 (ambiente, seeds, split por vídeo/cena)
- [ ] Wenndisson executa o Bloco 2 (métricas expandidas, IC 95%, latência)
- [ ] Wenndisson executa os Blocos 3 e 4 (robustez e validação cross-dataset), em paralelo se possível
- [ ] Decidir o recorte territorial e responder às sete perguntas de governança
- [ ] Escrever a seção de ética e LGPD (avaliação de impacto, retenção, viés, supervisão humana) e revisar "predictive security"
- [ ] Definir quem escreve e consolidar a declaração CRediT dos sete autores
- [ ] Escrever a seção de originalidade, citando o preprint arXiv:2502.18555 e o que o capítulo acrescenta
- [ ] Revisão final: o texto separa o que foi feito do que é só proposto e mantém a governança como tema central
