# **Plano de Revisão e Delegação de Simulações (Capítulo IGI Global)**

## **Veredito Geral**

Com base nos apontamentos editoriais (Exigências 1 e 3), o foco do trabalho deve ser a **consolidação rigorosa e o aprofundamento do modelo base (MobileNetV2 \+ BiLSTM)**. As frentes secundárias (Vision Transformers, Edge AI, Não-supervisionado, Grad-CAM e Dashboards de BI) foram cortadas do escopo prático para evitar dispersão metodológica. As simulações técnicas serão delegadas ao **Wenndisson**, que detém acesso ao ambiente de servidores.

## **Bloco 1: Preparação do Ambiente e Reprodutibilidade**

* **Escopo:** Configuração da pipeline de execução no servidor, fixação de *seeds* aleatórias para reprodutibilidade e criação de scripts para extração automática de métricas.

* **Tarefas:**

  1. Configurar scripts de treino, validação e teste com registro automático do ambiente (CUDA, framework, learning rate, batch size, épocas).

  2. Implementar *data split* estrito (70/15/15 ou 80/10/10) isolado no nível de **vídeo/cena** para evitar vazamento de dados (*data leakage*).

* **Estimativa de Tempo:** 1 a 2 dias úteis.

## **Bloco 2: Re-execução e Expansão de Métricas**

* **Escopo:** Reavaliação completa do modelo principal para extrair indicadores além da acurácia global.

* **Tarefas:**

  1. Extrair Precisão, Recall, Especificidade e F1-Score por classe e ponderado.

  2. Gerar Matriz de Confusão, Taxa de Falso Alerta (FPR) e Intervalos de Confiança (95%) via *bootstrapping*.

  3. Medir a Latência de Inferência (ms/frame) e *throughput* (FPS) no hardware do servidor.

* **Estimativa de Tempo:** 1 a 3 dias úteis.

## **Bloco 3: Testes de Robustez e Estresse do Modelo**

* **Escopo:** Avaliação da resiliência do modelo em cenários reais adversos conforme exigido pelo comitê editorial.

* **Tarefas:**

  1. **Iluminação:** Simular variações sintéticas de brilho e contraste (ambientes noturnos ou de alta exposição).

  2. **Oclusão e Aglomeração:** Aplicar máscaras de oclusão parcial e testar sequências em alta densidade populacional.

  3. **Qualidade de Câmera:** Testar degradação por compressão de vídeo e ruído (simulando CFTV público).

* **Estimativa de Tempo:** 3 a 5 dias úteis.

## **Bloco 4: Validação Externa (*Cross-Dataset*)**

* **Escopo:** Avaliação *zero-shot* do modelo em uma base secundária para demonstrar capacidade de generalização sem *overfitting*.

* **Tarefas:**

  1. Selecionar e pré-processar um dataset público de referência (ex.: *RLVS*, *UCF Crime* ou *Hockey Fight*) para a entrada de $100\times 100$ pixels.

  2. Executar inferências com o modelo treinado e consolidar a tabela comparativa de desempenho.

* **Estimativa de Tempo:** 3 a 4 dias úteis.

## **Bloco 5: Cronograma de Execução e Consolidação**

| Etapa | Responsável | Escopo Principal | Prazo Estimado |
| :---- | :---- | :---- | :---- |
| **Alinhamento & Dataset** | Erick / Liderança | Envio do código e base original | Dependência imediata |
| **Bloco 1: Setup & Pipeline** | Wenndisson | Logs de ambiente, seeds e scripts | 1 – 2 dias úteis |
| **Bloco 2: Métricas Expandidas** | Wenndisson | F1, FPR, Latência e Intervalos de Confiança | 1 – 3 dias úteis |
| **Bloco 3: Testes de Robustez** | Wenndisson | Perturbações de iluminação, oclusão e ruído | 3 – 5 dias úteis |
| **Bloco 4: Validação Externa** | Wenndisson | Testes no dataset público secundário | 3 – 4 dias úteis |
| **Integração & Redação** | Equipe de Redação | Redação das seções e geração de tabelas/gráficos | Paralelo/Sequencial |

