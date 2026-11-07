# Execução dos experimentos no servidor

Este fluxo prepara o RLVS, treina e avalia uma **reconstrução proposta** MobileNetV2 + BiLSTM + atenção em PyTorch. O artigo informa a combinação de componentes e entrada de 15 quadros RGB de 100 × 100; como os pesos, o código e detalhes do protocolo original não foram recebidos, esta implementação **não é uma reprodução comprovada**. Não há resultados científicos incluídos no repositório.

## 1. Ambiente e dados

Na raiz do projeto no servidor, instale os grupos e verifique o PyTorch:

```bash
uv sync --all-groups
uv run --no-sync python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

O arquivo `configs/rlvs_baseline.toml` usa `device = "cuda"`. Se a verificação retornar `False`, o treino com esse padrão falhará sem mudar silenciosamente para CPU. Instale/configure no servidor uma versão do PyTorch compatível com o driver e a GPU, seguindo as instruções oficiais do [PyTorch](https://pytorch.org/get-started/locally/) e do [uv para PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/). Depois de configurar o ambiente, os comandos abaixo usam `--no-sync` para não substituir a instalação escolhida. Para uso explícito em CPU, altere a configuração ou passe `--device cpu`; treinar em CPU pode ser demorado.

Configure o acesso ao Kaggle e baixe o RLVS como explicado no [guia de uso](../GUIA_DE_USO.md#2-baixar-o-dataset). O script fixa a versão 1 em `data/raw/rlvs-v1/`. Consulte as condições de uso, confirme se essa é a versão original e registre a fonte usada na pesquisa. A primeira construção do modelo com `pretrained = true` baixa pesos ImageNet do TorchVision, caso não estejam no cache do servidor; isso **não** são os pesos do modelo de conflito do trabalho anterior.

## 2. Configuração

Revise `configs/rlvs_baseline.toml` antes da execução. Os valores de épocas, lote, taxa de aprendizado, dropout, unidades, limiar, split e perturbações são parâmetros explícitos da reconstrução, não valores confirmados do experimento original. O limiar é fixado no arquivo antes do teste; não há calibração automática. Seeds registradas reduzem variação, mas não garantem determinismo entre GPUs, bibliotecas e sistemas.

O inventário reconhece arquivos de vídeo em diretórios chamados `Violence` e `NonViolence` (ignorando maiúsculas e espaços). Ele agrupa por SHA-256 e divide cada classe por grupos em treino, validação e teste, sem compartilhar um arquivo idêntico entre partições. Vídeos reeditados ou gravados na mesma cena podem ter hashes diferentes: revise essas relações e a licença antes da análise científica. A proporção final pode variar por causa do número de grupos. A divisão não garante independência por cena ou canal de origem.

## 3. Preparar dados

```bash
uv run --no-sync python scripts/research.py prepare --config configs/rlvs_baseline.toml
```

O comando identifica vídeos, classes e hashes e grava `data/processed/rlvs-baseline.json`, sem sobrescrever um manifesto existente. Ele não altera os arquivos originais. Revise as contagens por classe e as atribuições no manifesto; o notebook `04_prepare_rlvs.ipynb` mostra o resumo. Se o dataset usar nomes diferentes de diretórios ou outra organização, a preparação falhará e a adaptação deverá ser revisada explicitamente.

A leitura decodifica o vídeo inteiro, redimensiona para 100 × 100 e seleciona 15 quadros uniformemente ao longo dele; vídeos mais curtos repetem índices. É uma escolha documentada desta implementação e precisa ser comparada ao pré-processamento original. O método foi pensado para clipes curtos e pode consumir muita memória em vídeos longos.

## 4. Treinar

```bash
uv run --no-sync python scripts/research.py train --config configs/rlvs_baseline.toml
```

O comando informa um ID como `rlvs-...`. Para o mesmo fluxo no Jupyter, use `05_train_rlvs.ipynb` **no servidor**. O treino confere os hashes do manifesto, usa apenas treino/validação e seleciona `best.pt` pela menor perda de validação. Salva configuração efetiva, manifesto dos dados, histórico, gráfico, ambiente, revisão Git disponível e estado da execução em `output/<RUN_ID>/`. O conjunto de teste não é usado para selecionar o checkpoint. Uma falha deixa o estado `failed` no manifesto, em vez de marcar uma execução parcial como concluída.

## 5. Avaliar e medir robustez

Substitua `RUN_ID` pelo identificador informado no treinamento:

```bash
uv run --no-sync python scripts/research.py evaluate --run-id RUN_ID --robustness
```

O comando avalia uma sequência por vídeo da partição de teste e grava uma subpasta única em `output/<RUN_ID>/evaluations/`. O resultado inclui rótulos/predições, matriz, métricas por classe, FPR, intervalos por bootstrap agrupado e tempos de decodificação, pré-processamento e inferência. `--robustness` executa brilho, contraste, ruído, oclusão e compressão JPEG por quadro usando os níveis do TOML, com comparação pareada de acurácia. O notebook `06_evaluate_rlvs.ipynb` apresenta tabelas e gráficos.

Para incluir a compressão temporal do vídeo, instale **FFmpeg com suporte a libx264** no servidor e acrescente `--video-compression` junto de `--robustness`. Cada vídeo será recodificado temporariamente em H.264 com o CRF de `robustness.video_crf` antes da amostragem. A versão do FFmpeg é registrada; essa opção exige tempo e espaço temporário adicionais.

Os tempos são de processamento **offline** por sequência: não incluem aquisição de câmera ao vivo nem comprovam operação em tempo real. Perturbações são sintéticas; ruído independente entre quadros e máscaras de oclusão não demonstram desempenho em câmeras ou aglomerações reais. JPEG por quadro e recodificação H.264 são condições diferentes e não representam todos os codecs ou configurações de CFTV. A validade dos rótulos após as alterações deve ser inspecionada.

## 6. Validação externa

Escolha uma base de outra origem, com licença e rótulos compatíveis, e organize os vídeos em diretórios `Violence` e `NonViolence`. Copie o TOML de base para um novo arquivo em `configs/`, ajustando `data.root` e `data.manifest`. Informe uma origem distinta e crie o manifesto:

```bash
uv run --no-sync python scripts/research.py prepare --config configs/external_dataset.toml --dataset-name EXTERNAL_NAME --source EXTERNAL_SOURCE
uv run --no-sync python scripts/research.py evaluate --run-id RUN_ID --dataset-root PATH_TO_EXTERNAL_DATA --external-manifest PATH_TO_EXTERNAL_MANIFEST
```

O primeiro comando exige que o arquivo de configuração externo já tenha sido criado; os nomes em maiúsculas devem ser substituídos. A avaliação externa usa todos os registros desse manifesto, sem treinar nem ajustar o limiar. Ela recusa hashes idênticos aos da base original e fontes declaradas iguais. Isso não detecta reedições ou garante que uma base realmente seja independente ou nunca tenha sido usada no pré-treinamento: tais verificações são responsabilidade do protocolo. O notebook `07_external_evaluation.ipynb` usa a configuração externa já preparada.

## 7. Verificação e limites

As verificações locais em `tests/test_research_pipeline.py` usam arquivos temporários, inferência com pesos aleatórios e mocks das etapas de treino e avaliação; não treinam um modelo. O fluxo precisa de execução piloto no servidor com os dados e a GPU reais antes de ser considerado validado para os experimentos. Registre as decisões, verifique o trabalho anterior e não apresente os números sintéticos ou os parâmetros sugeridos como resultados originais.
