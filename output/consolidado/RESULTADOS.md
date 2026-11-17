# Resultados consolidados — protocolo fechado (RLVS → RWF-2000)

Gerado por `scripts/consolidate_results.py` a partir dos artefatos em `output/<RUN_ID>/`. Retreinamento em PyTorch a partir do ImageNet; **não** é o checkpoint `.keras` original.

## Protocolo

- 3 backbones (`mobilenet_v2`, `densenet121`, `inception_v3`) + BiLSTM + atenção; 15 quadros de 100 × 100; split por vídeo 0,64 / 0,16 / 0,20, seed 42 (1277 / 319 / 402 vídeos; 1998 após excluir um par idêntico com rótulos opostos).
- Limiar calibrado **na validação do RLVS** com FPR ≤ 0,05 (maximizando o recall) e congelado para teste, robustez e RWF-2000. A validação também escolheu o `best.pt`.
- IC 95%: bootstrap percentil reamostrando **vídeos** (grupos de conteúdo idêntico juntos), 1000 reamostragens. Classe positiva: Violence/Fight.

## Treino e limiar congelado

| backbone | run_id | epocas | melhor_epoca | acc_validacao | tempo_treino_s | limiar | fpr_validacao | recall_validacao |
|---|---|---|---|---|---|---|---|---|
| mobilenet_v2 | rlvs-20261116T182754Z-9f8e5bbb | 34 | 24 | 0,962 | 213 | 0,6709 | 0,050 | 0,987 |
| densenet121 | rlvs-20261116T182320Z-2c934236 | 44 | 34 | 0,956 | 307 | 0,3312 | 0,050 | 0,956 |
| inception_v3 | rlvs-20261116T182320Z-598ec442 | 43 | 33 | 0,915 | 261 | 0,9953 | 0,050 | 0,805 |

## Teste no RLVS (n = 402 vídeos) — estimativa [IC 95%]

| metrica | mobilenet_v2 | densenet121 | inception_v3 |
|---|---|---|---|
| Acurácia | 0,948 [0,925; 0,968] | 0,935 [0,908; 0,960] | 0,863 [0,828; 0,896] |
| Precisão (Violence) | 0,955 [0,922; 0,980] | 0,915 [0,872; 0,953] | 0,962 [0,929; 0,988] |
| Recall (Violence) | 0,940 [0,906; 0,971] | 0,960 [0,933; 0,986] | 0,756 [0,698; 0,810] |
| F1 (Violence) | 0,947 [0,923; 0,968] | 0,937 [0,910; 0,961] | 0,847 [0,806; 0,882] |
| Precisão (NonViolence) | 0,941 [0,906; 0,972] | 0,958 [0,931; 0,985] | 0,799 [0,747; 0,850] |
| Recall (NonViolence) | 0,955 [0,923; 0,980] | 0,910 [0,866; 0,951] | 0,970 [0,946; 0,990] |
| F1 (NonViolence) | 0,948 [0,925; 0,968] | 0,934 [0,904; 0,958] | 0,876 [0,842; 0,907] |
| Especificidade | 0,955 [0,923; 0,980] | 0,910 [0,866; 0,951] | 0,970 [0,946; 0,990] |
| F1 macro | 0,948 [0,925; 0,968] | 0,935 [0,908; 0,960] | 0,862 [0,827; 0,892] |
| FPR | 0,045 [0,020; 0,077] | 0,090 [0,049; 0,134] | 0,030 [0,010; 0,054] |

## Validação externa no RWF-2000 (n = 2000 vídeos, limiar congelado do RLVS, sem ajuste)

| metrica | mobilenet_v2 | densenet121 | inception_v3 |
|---|---|---|---|
| Acurácia | 0,575 [0,554; 0,598] | 0,542 [0,522; 0,565] | 0,558 [0,537; 0,581] |
| Precisão (Violence) | 0,570 [0,542; 0,599] | 0,537 [0,510; 0,566] | 0,560 [0,530; 0,592] |
| Recall (Violence) | 0,613 [0,581; 0,646] | 0,608 [0,577; 0,639] | 0,544 [0,513; 0,577] |
| F1 (Violence) | 0,591 [0,566; 0,615] | 0,570 [0,547; 0,597] | 0,552 [0,526; 0,580] |
| Precisão (NonViolence) | 0,581 [0,550; 0,615] | 0,548 [0,519; 0,582] | 0,556 [0,528; 0,590] |
| Recall (NonViolence) | 0,537 [0,507; 0,567] | 0,476 [0,446; 0,507] | 0,572 [0,543; 0,602] |
| F1 (NonViolence) | 0,558 [0,532; 0,585] | 0,510 [0,485; 0,536] | 0,564 [0,540; 0,589] |
| Especificidade | 0,537 [0,507; 0,567] | 0,476 [0,446; 0,507] | 0,572 [0,543; 0,602] |
| F1 macro | 0,574 [0,553; 0,597] | 0,540 [0,520; 0,563] | 0,558 [0,537; 0,580] |
| FPR | 0,463 [0,433; 0,493] | 0,524 [0,493; 0,554] | 0,428 [0,398; 0,457] |

## Matrizes de confusão

| backbone | conjunto | TN | FP | FN | TP | n |
|---|---|---|---|---|---|---|
| mobilenet_v2 | RLVS teste | 192 | 9 | 12 | 189 | 402 |
| mobilenet_v2 | RWF-2000 | 537 | 463 | 387 | 613 | 2000 |
| densenet121 | RLVS teste | 183 | 18 | 8 | 193 | 402 |
| densenet121 | RWF-2000 | 476 | 524 | 392 | 608 | 2000 |
| inception_v3 | RLVS teste | 195 | 6 | 49 | 152 | 402 |
| inception_v3 | RWF-2000 | 572 | 428 | 456 | 544 | 2000 |

## Robustez sintética no teste do RLVS (Δ = diferença pareada de acurácia vs. limpo, IC 95%)

| condicao | mobilenet_v2 | densenet121 | inception_v3 |
|---|---|---|---|
| Brilho | acc 0,940 (Δ -0,007 [-0,022; 0,007]); FPR 0,040 | acc 0,928 (Δ -0,007 [-0,025; 0,010]); FPR 0,095 | acc 0,838 (Δ -0,025 [-0,043; -0,007]); FPR 0,025 |
| Contraste | acc 0,948 (Δ 0,000 [-0,017; 0,015]); FPR 0,065 | acc 0,938 (Δ 0,002 [-0,010; 0,015]); FPR 0,085 | acc 0,843 (Δ -0,020 [-0,040; -0,002]); FPR 0,025 |
| Ruído gaussiano | acc 0,831 (Δ -0,117 [-0,155; -0,082]); FPR 0,045 | acc 0,888 (Δ -0,047 [-0,077; -0,017]); FPR 0,060 | acc 0,791 (Δ -0,072 [-0,102; -0,045]); FPR 0,020 |
| Oclusão central | acc 0,938 (Δ -0,010 [-0,025; 0,002]); FPR 0,060 | acc 0,682 (Δ -0,254 [-0,303; -0,201]); FPR 0,015 | acc 0,672 (Δ -0,192 [-0,236; -0,142]); FPR 0,060 |
| JPEG por quadro | acc 0,928 (Δ -0,020 [-0,042; 0,000]); FPR 0,114 | acc 0,891 (Δ -0,045 [-0,073; -0,015]); FPR 0,194 | acc 0,853 (Δ -0,010 [-0,037; 0,015]); FPR 0,045 |
| Recodificação H.264 | acc 0,945 (Δ -0,002 [-0,017; 0,012]); FPR 0,070 | acc 0,928 (Δ -0,007 [-0,022; 0,007]); FPR 0,119 | acc 0,871 (Δ 0,007 [-0,007; 0,025]); FPR 0,030 |

Perturbações sintéticas sobre vídeos observados; não representam câmeras ou aglomerações reais. JPEG por quadro e recodificação H.264 (libx264, CRF 28) são condições distintas. ICs completos de todas as métricas por condição estão em `robustness.json` de cada avaliação.

## Latência (RTX 4090, lote 1, 20 inferências de aquecimento)

| backbone | offline_total_mediana_ms | offline_total_p95_ms | offline_inferencia_mediana_ms | offline_inferencia_p95_ms | stream_decisao_mediana_ms | stream_decisao_p95_ms | stream_quadro_p95_ms | carga_tempo_real_p95 | atraso_janela_s | videos | decisoes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mobilenet_v2 | 41,1 | 222,3 | 1,89 | 2,40 | 2,08 | 2,98 | 2,69 | 0,083 | 2,0 | 100 | 560 |
| densenet121 | 45,7 | 232,6 | 5,15 | 5,79 | 5,02 | 6,23 | 2,72 | 0,091 | 2,0 | 100 | 560 |
| inception_v3 | 43,9 | 221,8 | 3,73 | 4,26 | 4,00 | 4,83 | 2,70 | 0,089 | 2,0 | 100 | 560 |

- **Offline:** clipe gravado inteiro — decodificação em duas passagens + pré-processamento + inferência. Não é atraso de alerta ao vivo.
- **Streaming (tempo real simulado):** cada vídeo reproduzido a 30 fps, janela de 2,0 s (60 quadros, 15 amostrados), decisão a cada 0,5 s. O alerta só pode sair depois de a janela se completar (atraso de formação = 2,0 s), mais o tempo por decisão. "Carga" = fração do intervalo entre decisões gasta com processamento (p95). Só o tempo é medido; a acurácia em janelas deslizantes não foi avaliada, e o modelo foi treinado com amostragem do clipe inteiro. O tempo de decodificação por quadro é amortizado (decodificador multithread do OpenCV) e não inclui captura/rede de uma câmera real.

## Hardware

- **Treino (GPU):** `mobilenet_v2` em work5 (NVIDIA GeForce RTX 4090, CPU AMD Ryzen 7 7700X 8-Core Processor, 125.5 GiB RAM); `densenet121` em work5 (NVIDIA GeForce RTX 4090, CPU AMD Ryzen 7 7700X 8-Core Processor, 125.5 GiB RAM); `inception_v3` em work5 (NVIDIA GeForce RTX 4090, CPU AMD Ryzen 7 7700X 8-Core Processor, 125.5 GiB RAM).
- **Latência (GPU):** work6 — NVIDIA GeForce RTX 4090 (23.5 GiB, CC 8.9, driver 610.43.02, CUDA 12.6, cuDNN 90501); CPU AMD Ryzen 7 7700X 8-Core Processor (8 núcleos físicos / 16 lógicos; 8 alocados ao job); RAM 61.9 GiB; 4 threads PyTorch; Linux-7.0.0-28-generic-x86_64-with-glibc2.39.
- **Calibração, teste, robustez e RWF-2000 (CPU, mesmo dispositivo para todos os scores):** work7 — CPU AMD Ryzen 9 7950X 16-Core Processor (16 físicos / 32 lógicos; 8 alocados); RAM 187.8 GiB; 8 threads PyTorch.
- **Software:** PyTorch 2.7.1+cu126, torchvision 0.22.1, OpenCV 4.14.0.94, Python 3.12.11; FFmpeg: ffmpeg version 9.0.2 Copyright (c) 2000-2026 the FFmpeg developers.

## Proveniência

- `mobilenet_v2`: run `rlvs-20261116T182754Z-9f8e5bbb`; teste `output/rlvs-20261116T182754Z-9f8e5bbb/evaluations/20261116T201515Z-8b86db21`; RWF `output/rlvs-20261116T182754Z-9f8e5bbb/evaluations/20261116T202821Z-28d85b8d`; limiar `output/rlvs-20261116T182754Z-9f8e5bbb/threshold.json`.
- `densenet121`: run `rlvs-20261116T182320Z-2c934236`; teste `output/rlvs-20261116T182320Z-2c934236/evaluations/20261116T201520Z-1224df2d`; RWF `output/rlvs-20261116T182320Z-2c934236/evaluations/20261116T202939Z-04270d4d`; limiar `output/rlvs-20261116T182320Z-2c934236/threshold.json`.
- `inception_v3`: run `rlvs-20261116T182320Z-598ec442`; teste `output/rlvs-20261116T182320Z-598ec442/evaluations/20261116T201515Z-1800a77c`; RWF `output/rlvs-20261116T182320Z-598ec442/evaluations/20261116T202821Z-decf2476`; limiar `output/rlvs-20261116T182320Z-598ec442/threshold.json`.
