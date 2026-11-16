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
- **Modelo (`[model]`):** `backbone` selecionável entre os 3 originais (`"mobilenet_v2"`, `"densenet121"`, `"inception_v3"`) e 5 arquiteturas modernas/Vision Transformers (`"mobilenet_v3_large"`, `"efficientnet_v2_s"`, `"convnext_tiny"`, `"swin_t"`, `"vit_b_16"`), `pretrained = true`, `trainable_layers = 80` (libera os últimos 80 módulos folha), `pooling = "avg"` (ou `"flatten"`), `temporal_head = "bilstm"` (ou `"transformer"`), `hidden_size = 32`, `attention = true` (ou `false` para experimentos sem atenção), `dense_units = [256, 128, 64, 32]` e `dropout = 0.25`.
- **Treinamento (`[training]`):** `epochs = 100`, `batch_size = 64` (ou `128`), `optimizer = "sgd"`, `learning_rate = 0.01`, `momentum = 0.9`, `ReduceLROnPlateau` monitorando `validation_loss` (`lr_factor = 0.6`, `lr_patience = 5`, `min_learning_rate = 0.00005` ou `0.0005`) e `EarlyStopping` monitorando `validation_accuracy` (`early_stopping_patience = 10`).

O limiar de decisão é **calibrado na validação do RLVS** pelo comando `calibrate` (seção 4.1), com teto de FPR `evaluation.target_fpr = 0.05`, e congelado em `output/<RUN_ID>/threshold.json`. O mesmo limiar é usado no teste do RLVS, na robustez e no RWF-2000; o `evaluate` recusa rodar sem esse arquivo. `evaluation.threshold = 0.5` é só valor de referência e não decide nenhuma avaliação. Seeds registradas reduzem variação, mas não garantem determinismo entre GPUs, bibliotecas e sistemas.

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

### 4.1 Calibrar e congelar o limiar

```bash
uv run --no-sync python scripts/research.py calibrate --run-id RUN_ID
```

O comando pontua a partição de validação com o `best.pt` e escolhe o limiar de menor margem que mantém o FPR da validação ≤ `target_fpr`, o que maximiza o recall sob esse teto. O limiar fica no ponto médio entre o maior score negativo rejeitado e o score seguinte, e a classe positiva vale para `score >= limiar`. O resultado vai para `threshold.json`, vinculado ao SHA-256 do `best.pt`, junto com as predições da validação em `calibration/`. A calibração não pode ser refeita para o mesmo run. A validação também escolheu o `best.pt`, por isso o FPR em dados novos pode passar da meta; isso deve ser declarado.

## 5. Avaliar e medir robustez

Substitua `RUN_ID` pelo identificador informado no treinamento:

```bash
uv run --no-sync python scripts/research.py evaluate --run-id RUN_ID --robustness
```

O comando avalia uma sequência por vídeo da partição de teste e grava uma subpasta única em `output/<RUN_ID>/evaluations/`. O resultado inclui rótulos/predições, matriz, métricas por classe, FPR, intervalos por bootstrap agrupado e tempos de decodificação, pré-processamento e inferência. `--robustness` executa brilho, contraste, ruído, oclusão e compressão JPEG por quadro usando os níveis do TOML, com comparação pareada de acurácia. O notebook `03_evaluate_test.ipynb` apresenta tabelas e gráficos.

Para incluir a compressão temporal do vídeo, instale **FFmpeg com suporte a libx264** no servidor e acrescente `--video-compression` junto de `--robustness`. Cada vídeo será recodificado temporariamente em H.264 com o CRF de `robustness.video_crf` antes da amostragem. A versão do FFmpeg é registrada; essa opção exige tempo e espaço temporário adicionais.

Os tempos são de processamento **offline** por sequência: não incluem aquisição de câmera ao vivo nem comprovam operação em tempo real. Perturbações são sintéticas; ruído independente entre quadros e máscaras de oclusão não demonstram desempenho em câmeras ou aglomerações reais. JPEG por quadro e recodificação H.264 são condições diferentes e não representam todos os codecs ou configurações de CFTV. A validade dos rótulos após as alterações deve ser inspecionada.

## 6. Validação externa (*cross-dataset*)

Para uma comparação externa justa com o RLVS (2.000 vídeos), o projeto já inclui a configuração `configs/rwf2000_external.toml` apontando para o benchmark **RWF-2000** (*Real-World Fighting Dataset*, `vulamnguyen/rwf2000` no Kaggle: 2.000 vídeos de 5 s capturados por câmeras de vigilância reais, divididos igualmente em `Fight` e `NonFight`). O inventário reconhece automaticamente diretórios `Fight`/`NonFight` (RWF-2000 e Hockey Fight) e `Normal`/`Violence` (SCVD), além de `Violence`/`NonViolence` (RLVS):

```bash
uv run --group data python scripts/download_dataset.py --dataset rwf2000
uv run --no-sync python scripts/research.py prepare --config configs/rwf2000_external.toml
uv run --no-sync python scripts/research.py evaluate --run-id RUN_ID --dataset-root data/raw/rwf2000-v1 --external-manifest data/processed/rwf2000-external.json
```

A avaliação externa usa todos os registros desse manifesto, sem treinar nem ajustar o limiar. Ela recusa hashes idênticos aos da base original e fontes declaradas iguais. Isso não detecta reedições ou garante que uma base nunca tenha sido compartilhada com o RLVS: tais verificações continuam sob responsabilidade do protocolo. O notebook `04_evaluate_external.ipynb` já aponta para `configs/rwf2000_external.toml`.

## 6.1 Latência (offline e streaming)

```bash
uv run --no-sync python scripts/research.py latency --run-id RUN_ID --device cuda
```

O comando mede duas coisas, separadamente, em até 100 vídeos de teste, com 20 inferências de aquecimento e lote 1:

- **Offline:** decodifica o clipe inteiro em duas passagens, faz o pré-processamento e a inferência. O clipe precisa estar gravado antes.
- **Streaming:** reproduz cada vídeo como se fosse ao vivo, a 30 fps, com janela de 2 s, passo de 0,5 s e 15 quadros por janela. Mede a decodificação e o redimensionamento por quadro, o tempo por decisão e o orçamento de tempo real.

Só o tempo é medido; a acurácia em janelas deslizantes não é avaliada. O hardware (GPU, driver, CPU, RAM, threads) é gravado em `output/<RUN_ID>/latency/<ID>/latency.json`.

## 6.2 Execução no Heisenberg

Os scripts em `jobs/` reservam uma RTX 4090 (`-p gpu --gres=gpu:4090:1`), usam o `/tmp` local do nó como `TMPDIR` e o apagam no fim. O ffmpeg vem de `~/Packages/ffmpeg-env`. Os comandos são enviados a partir da raiz do repositório:

```bash
sbatch -J train-densenet121 --export=ALL,BACKBONE=densenet121 jobs/train.slurm
sbatch -J eval-densenet121 --export=ALL,RUN_ID=rlvs-... jobs/evaluate.slurm   # calibrate -> teste+robustez+H.264 -> RWF-2000
sbatch --export=ALL,RUN_IDS="rlvs-a rlvs-b rlvs-c" jobs/latency.slurm          # work5 exclusivo
```

O RLVS do Kaggle (v1) traz duas cópias idênticas da base. A raiz aponta para `data/raw/rlvs-v1/Real Life Violence Dataset`. A base tem 13 pares de vídeos idênticos na mesma classe, que ficam agrupados na mesma partição, e um par idêntico em classes opostas (`NonViolence/NV_226.mp4` = `Violence/V_504.mp4`). Esse par é excluído com `prepare --drop-label-conflicts` e registrado no manifesto. Com seed 42, a divisão fica em 1277 / 319 / 402 vídeos, num total de 1998.

## 7. Verificação e limites

As verificações locais em `tests/test_research_pipeline.py` usam arquivos temporários, inferência com pesos aleatórios e mocks das etapas de treino e avaliação; não treinam um modelo. O fluxo precisa de execução piloto no servidor com os dados e a GPU reais antes de ser considerado validado para os experimentos. Registre as decisões, verifique o trabalho anterior e não apresente os números sintéticos ou os parâmetros sugeridos como resultados originais.
