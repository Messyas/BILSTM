# Execução dos experimentos no servidor

Este fluxo prepara o RLVS, treina e avalia a arquitetura CNN + BiLSTM + atenção alinhada aos notebooks originais enviados por Erick Farias (`FEATURE_EXTRACTION.ipynb` e `parametro_alterado_definitivo_densenet121_att(1).ipynb`). O código original foi escrito em TensorFlow/Keras para TPU no Google Colab; esta implementação adapta a arquitetura, a extração de sequências matriciais de 15 quadros RGB de 100 × 100 e os hiperparâmetros de treinamento para o pipeline reproduzível em PyTorch, pois os arquivos de pesos `.keras` treinados originalmente não foram anexados. Não há resultados científicos pré-calculados incluídos no repositório.

## 1. Ambiente e dados

Na raiz do projeto no servidor, instale os grupos e verifique o PyTorch:

```bash
uv sync --all-groups
uv run --no-sync python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

O arquivo `configs/rlvs_baseline.toml` usa `device = "cuda"`. Se a verificação retornar `False`, o treino com esse padrão falhará sem mudar silenciosamente para CPU. Instale/configure no servidor uma versão do PyTorch compatível com o driver e a GPU, seguindo as instruções oficiais do [PyTorch](https://pytorch.org/get-started/locally/) e do [uv para PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/). Depois de configurar o ambiente, os comandos abaixo usam `--no-sync` para não substituir a instalação escolhida. Para uso explícito em CPU, altere a configuração ou passe `--device cpu`; treinar em CPU pode ser demorado.

Configure o acesso ao Kaggle e baixe o RLVS como explicado no [guia de uso](../GUIA_DE_USO.md#2-baixar-o-dataset). O script fixa a versão 1 em `data/raw/rlvs-v1/`. Consulte as condições de uso, confirme se essa é a versão original e registre a fonte usada na pesquisa. A primeira construção do modelo com `pretrained = true` baixa pesos ImageNet do TorchVision, caso não estejam no cache do servidor; isso **não** são os pesos do modelo de conflito do trabalho anterior.

## 2. Configuração

Revise `configs/rlvs_baseline.toml` antes da execução. Os parâmetros reproduzem a configuração identificada nos notebooks originais de Erick Farias:

- **Divisão (`[split]`):** `train = 0.64`, `validation = 0.16`, `test = 0.20` e `seed = 42` (nos 2000 vídeos do RLVS, equivale a 1280 vídeos de treino, 320 de validação e 400 de teste, obtida originalmente com `test_size = 0.20` e `validation_split = 0.20`).
- **Sequência de vídeo (`[video]`):** `frames = 15`, `width = 100`, `height = 100`, amostrados com passo inteiro `total_frames // 15` e normalizados em `[0, 1]` (`/ 255`).
- **Modelo (`[model]`):** `backbone` selecionável entre `"mobilenet_v2"` (padrão), `"densenet121"` e `"inception_v3"`, `pretrained = true`, `trainable_layers = 80` (libera os últimos 80 módulos folha da CNN), `pooling = "avg"` (ou `"flatten"`), `hidden_size = 32` na BiLSTM, `attention = true` (ou `false` para experimentos sem atenção), `dense_units = [256, 128, 64, 32]` e `dropout = 0.25`.
- **Treinamento (`[training]`):** `epochs = 100`, `batch_size = 64` (ou `128`), `optimizer = "sgd"`, `learning_rate = 0.01`, `momentum = 0.9`, `ReduceLROnPlateau` monitorando `validation_loss` (`lr_factor = 0.6`, `lr_patience = 5`, `min_learning_rate = 0.00005` ou `0.0005`) e `EarlyStopping` monitorando `validation_accuracy` (`early_stopping_patience = 10`).

O limiar de decisão (`evaluation.threshold = 0.5`) é fixado no arquivo antes do teste; não há calibração automática. Seeds registradas reduzem variação, mas não garantem determinismo entre GPUs, bibliotecas e sistemas.

O inventário reconhece arquivos de vídeo em diretórios chamados `Violence` e `NonViolence` (ignorando maiúsculas e espaços). Por padrão ele agrupa por SHA-256 e divide cada classe por grupos em treino, validação e teste, sem compartilhar um arquivo idêntico entre partições. Vídeos reeditados ou gravados na mesma cena podem ter hashes diferentes: revise essas relações e a licença antes da análise científica. A proporção final pode variar por causa do número de grupos. A divisão padrão não garante independência por cena ou canal de origem.

## 3. Preparar dados

```bash
uv run --no-sync python scripts/research.py prepare --config configs/rlvs_baseline.toml
```

O comando identifica vídeos, classes e hashes e grava `data/processed/rlvs-baseline.json`, sem sobrescrever um manifesto existente. Ele não altera os arquivos originais. Revise as contagens por classe e as atribuições no manifesto; o notebook `01_prepare_dataset.ipynb` mostra o resumo. Se o dataset usar nomes diferentes de diretórios ou outra organização, a preparação falhará e a adaptação deverá ser revisada explicitamente.

Quando houver metadados revisados, é possível informar um CSV UTF-8 com cabeçalho `sample_id,group_id`, contendo exatamente uma linha para cada vídeo. `sample_id` é o caminho relativo à raiz do dataset, com `/`; use o mesmo `group_id` para vídeos da mesma cena ou origem que não possam cruzar partições. Por exemplo:

```csv
sample_id,group_id
Violence/video1.mp4,scene-1
Violence/video2.mp4,scene-1
```

O exemplo é ilustrativo e não representa um mapeamento completo. Com o CSV completo preparado e revisado, informe `--groups caminho/grupos.csv` ao comando `prepare`. O manifesto registra o hash do CSV e os grupos; IDs ausentes ou extras, amostras repetidas e arquivos idênticos em grupos distintos são recusados. O algoritmo atual exige que todos os vídeos de um grupo tenham a mesma classe e pelo menos três grupos por classe; se uma cena contiver ambas as classes, será necessário definir e implementar outro protocolo de divisão. Não se deve separar a cena artificialmente para contornar essa restrição. O CSV só protege contra relações nele identificadas; a equipe deve revisar sua completude.

A leitura faz duas passagens sequenciais pelo vídeo: conta os quadros decodificáveis e retém os 15 selecionados pelo intervalo inteiro `total_frames // 15` (ou espaçamento uniforme se o vídeo tiver menos de 15 quadros), convertendo de BGR para RGB, redimensionando para 100 × 100 e escalando em `[0, 1]`. No notebook original de extração (`FEATURE_EXTRACTION.ipynb`), havia aumentos aleatórios (`imgaug` de brilho, zoom e desfoque gaussiano) aplicados durante a extração de todos os vídeos antes da divisão; neste pipeline a extração de treino, validação e teste é determinística e sem aumento prévio ao split para preservar a integridade da avaliação, mantendo as perturbações isoladas na etapa de robustez.

**Limite temporal:** a amostragem usa o vídeo inteiro e a BiLSTM bidirecional usa os quadros posteriores da sequência. A previsão é uma classificação offline do clipe completo, não uma detecção causal no instante de cada quadro nem um teste de antecipação. Cada chamada cria o estado recorrente da sequência sem carregá-lo para o próximo vídeo. Para aplicação ao vivo seriam necessários janelas disponíveis até o instante de decisão, regras de agregação e medição do atraso de aquisição.

## 4. Treinar

```bash
uv run --no-sync python scripts/research.py train --config configs/rlvs_baseline.toml
```

Para variar o modelo ou hiperparâmetros (como nos 12 experimentos da Tabela 1 do artigo original), edite o TOML ou informe opções explícitas na linha de comando, por exemplo:

```bash
uv run --no-sync python scripts/research.py train --config configs/rlvs_baseline.toml --backbone densenet121 --attention --batch-size 64 --min-learning-rate 0.00005
```

O comando informa um ID como `rlvs-...`. Para o mesmo fluxo no Jupyter, use `02_train_baseline.ipynb` **no servidor**. O treino confere os hashes do manifesto, converte e armazena em cache as sequências matriciais de treino e validação na primeira época, aplica `ReduceLROnPlateau` na perda de validação e `EarlyStopping` com seleção de `best.pt` pela maior acurácia de validação (desempatando pela menor perda de validação). Salva configuração efetiva, manifesto dos dados, histórico de perda/acurácia/taxa de aprendizado, gráfico, tempo de treinamento, ambiente, revisão Git disponível e estado da execução em `output/<RUN_ID>/`. O conjunto de teste não é usado para selecionar o checkpoint. Uma falha deixa o estado `failed` no manifesto, em vez de marcar uma execução parcial como concluída.

## 5. Avaliar e medir robustez

Substitua `RUN_ID` pelo identificador informado no treinamento:

```bash
uv run --no-sync python scripts/research.py evaluate --run-id RUN_ID --robustness
```

O comando avalia uma sequência por vídeo da partição de teste e grava uma subpasta única em `output/<RUN_ID>/evaluations/`. O resultado inclui rótulos/predições, matriz, métricas por classe, FPR, intervalos por bootstrap agrupado e tempos de decodificação, pré-processamento e inferência. `--robustness` executa brilho, contraste, ruído, oclusão e compressão JPEG por quadro usando os níveis do TOML, com comparação pareada de acurácia. O notebook `03_evaluate_test.ipynb` apresenta tabelas e gráficos.

Para incluir a compressão temporal do vídeo, instale **FFmpeg com suporte a libx264** no servidor e acrescente `--video-compression` junto de `--robustness`. Cada vídeo será recodificado temporariamente em H.264 com o CRF de `robustness.video_crf` antes da amostragem. A versão do FFmpeg é registrada; essa opção exige tempo e espaço temporário adicionais.

Os tempos são de processamento **offline** por sequência: não incluem aquisição de câmera ao vivo nem comprovam operação em tempo real. Perturbações são sintéticas; ruído independente entre quadros e máscaras de oclusão não demonstram desempenho em câmeras ou aglomerações reais. JPEG por quadro e recodificação H.264 são condições diferentes e não representam todos os codecs ou configurações de CFTV. A validade dos rótulos após as alterações deve ser inspecionada.

## 6. Validação externa

Escolha uma base de outra origem, com licença e rótulos compatíveis, e organize os vídeos em diretórios `Violence` e `NonViolence`. Copie o TOML de base para um novo arquivo em `configs/`, ajustando `data.root` e `data.manifest`. Informe uma origem distinta e crie o manifesto:

```bash
uv run --no-sync python scripts/research.py prepare --config configs/external_dataset.toml --dataset-name EXTERNAL_NAME --source EXTERNAL_SOURCE
uv run --no-sync python scripts/research.py evaluate --run-id RUN_ID --dataset-root PATH_TO_EXTERNAL_DATA --external-manifest PATH_TO_EXTERNAL_MANIFEST
```

O primeiro comando exige que o arquivo de configuração externo já tenha sido criado; os nomes em maiúsculas devem ser substituídos. A avaliação externa usa todos os registros desse manifesto, sem treinar nem ajustar o limiar. Ela recusa hashes idênticos aos da base original e fontes declaradas iguais. Isso não detecta reedições ou garante que uma base realmente seja independente ou nunca tenha sido usada no pré-treinamento: tais verificações são responsabilidade do protocolo. O notebook `04_evaluate_external.ipynb` usa a configuração externa já preparada.

## 7. Verificação e limites

As verificações locais em `tests/test_research_pipeline.py` usam arquivos temporários, inferência com pesos aleatórios e mocks das etapas de treino e avaliação; não treinam um modelo. O fluxo precisa de execução piloto no servidor com os dados e a GPU reais antes de ser considerado validado para os experimentos. Registre as decisões, verifique o trabalho anterior e não apresente os números sintéticos ou os parâmetros sugeridos como resultados originais.
