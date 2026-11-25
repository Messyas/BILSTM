# Modelos treinados (protocolo fechado, Heisenberg, 16/11/2026)

São 18 modelos: 3 backbones × 6 seeds de treino (1–5 e 42). Todos usam BiLSTM + atenção e foram treinados com o mesmo split por vídeo do RLVS. Os três modelos de seed 42 são os principais do capítulo. Os outros 15 são o adendo de estabilidade (`consolidado/suplementar_seeds/`).

## Arquivos de cada run (`output/<run_id>/`)

| arquivo | conteúdo |
|---|---|
| `best.pt` | pesos do PyTorch (`{"model_state": state_dict, ...}`) da época com a melhor acurácia de validação |
| `effective_config.toml` | arquitetura e parâmetros exatos do run (`[model]`, `[video]`) |
| `threshold.json` | limiar congelado, calibrado na validação com meta de FPR ≤ 0,05, mais o sha256 do `best.pt` |
| `manifest.json`, `history.json` | estado do run e curva de treino por época |

## Lista dos modelos

| backbone | seed | run_id | best.pt | limiar | acurácia teste RLVS | FPR teste |
|---|---|---|---|---|---|---|
| densenet121 | 42 (principal) | `rlvs-20261116T182320Z-2c934236` | 29.7 MB | 0.3312 | 0,935 | 0,090 |
| densenet121 | 1 | `rlvs-20261116T203129Z-d36e0ce2` | 29.7 MB | 0.8508 | 0,943 | 0,030 |
| densenet121 | 2 | `rlvs-20261116T220234Z-707706ca` | 29.7 MB | 0.1137 | 0,940 | 0,080 |
| densenet121 | 3 | `rlvs-20261116T210114Z-921c9228` | 29.7 MB | 0.2823 | 0,943 | 0,080 |
| densenet121 | 4 | `rlvs-20261116T211954Z-d26672d7` | 29.7 MB | 0.9743 | 0,933 | 0,055 |
| densenet121 | 5 | `rlvs-20261116T214447Z-53bb7185` | 29.7 MB | 0.2856 | 0,938 | 0,080 |
| inception_v3 | 42 (principal) | `rlvs-20261116T182320Z-598ec442` | 89.8 MB | 0.9953 | 0,863 | 0,030 |
| inception_v3 | 1 | `rlvs-20261116T203734Z-dedba4c2` | 89.8 MB | 0.9567 | 0,895 | 0,050 |
| inception_v3 | 2 | `rlvs-20261116T204844Z-2d5c6990` | 89.8 MB | 0.5870 | 0,918 | 0,065 |
| inception_v3 | 3 | `rlvs-20261116T210753Z-9f13ce26` | 89.8 MB | 0.9892 | 0,866 | 0,050 |
| inception_v3 | 4 | `rlvs-20261116T213134Z-61e5ee11` | 89.8 MB | 0.9978 | 0,873 | 0,020 |
| inception_v3 | 5 | `rlvs-20261116T215102Z-d379362d` | 89.8 MB | 0.7956 | 0,918 | 0,055 |
| mobilenet_v2 | 42 (principal) | `rlvs-20261116T182754Z-9f8e5bbb` | 10.7 MB | 0.6709 | 0,948 | 0,045 |
| mobilenet_v2 | 1 | `rlvs-20261116T215621Z-3e5a9666` | 10.7 MB | 0.9857 | 0,953 | 0,040 |
| mobilenet_v2 | 2 | `rlvs-20261116T204311Z-98d34a68` | 10.7 MB | 0.7609 | 0,958 | 0,035 |
| mobilenet_v2 | 3 | `rlvs-20261116T205440Z-7c99e3c4` | 10.7 MB | 0.4692 | 0,960 | 0,045 |
| mobilenet_v2 | 4 | `rlvs-20261116T211355Z-fcf46d8a` | 10.7 MB | 0.7414 | 0,945 | 0,050 |
| mobilenet_v2 | 5 | `rlvs-20261116T213835Z-a5510e6f` | 10.7 MB | 0.7386 | 0,968 | 0,030 |

## Como carregar (Python, a partir da raiz do repositório)

```python
import torch
from bilstm_pipeline.research_evaluation import _load_run, load_frozen_threshold

run_dir, config, device, model = _load_run("rlvs-20261116T182754Z-9f8e5bbb", "cpu")  # mobilenet_v2, seed 42
threshold = load_frozen_threshold(run_dir)  # confere o sha256 do best.pt
videos = torch.rand(1, 15, 3, 100, 100)     # [lote, quadros, canais RGB, altura, largura], valores em [0, 1]
with torch.inference_mode():
    p_violence = model(videos).double().softmax(1)[0, 1].item()
violence = p_violence >= threshold["threshold"]
```

O modelo também pode ser montado direto com `ConflictModel(**config["model"], pretrained=False, **config["video"])` seguido de `load_state_dict(torch.load("best.pt")["model_state"])`.

## Entrada e saída

**Entrada:** 15 quadros por vídeo, preparados assim:
- índices `i × (N // 15)`, para `i = 0…14`, onde N é o número de quadros do vídeo;
- conversão de BGR para RGB;
- redimensionamento para 100×100 com `cv2.INTER_AREA`;
- divisão por 255. **Não há normalização do ImageNet.**

**Saída:** 2 logits, na ordem `[NonViolence, Violence]`. Aplica-se o softmax ao segundo, e o vídeo é classificado como violência quando o resultado é **≥ o limiar do `threshold.json` do próprio run**. Cada seed tem um limiar diferente, que não deve ser trocado por 0,5.

## Embarcar (microcontrolador ou edge)

**Só a mobilenet_v2 é viável.** Ela ocupa ~10,7 MB em float32, ~2,7 M parâmetros. A densenet121 ocupa 30 MB e a inception_v3, 90 MB.

O caminho típico:
1. Exportar com `torch.onnx.export`, com entrada `[1, 15, 3, 100, 100]`.
2. Converter para o runtime do alvo (TFLite Micro, ST Edge AI, ESP-DL etc.), quantizando em int8 com vídeos da **validação** como dados de calibração.

A quantização muda os scores. Depois dela é preciso:
- medir de novo a acurácia e o FPR no **teste**;
- recalibrar o limiar na **validação**, senão a meta de FPR deixa de valer.

As latências deste repositório (`consolidado/latencia.csv`) foram medidas numa RTX 4090 e não valem para microcontroladores.
