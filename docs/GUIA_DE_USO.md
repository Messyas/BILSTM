# Guia de uso do projeto

Este guia define o fluxo e os lugares do projeto para que os experimentos permaneçam fáceis de encontrar e reproduzir. Antes de criar uma pasta nova, use os caminhos deste guia.

## 1. Preparar o ambiente

Instale o uv seguindo a documentação oficial. No Windows, abra o PowerShell na raiz do projeto e rode:

```powershell
uv sync --all-groups
uv run --group notebooks python -m ipykernel install --user --name bilstm-experimentos --display-name "Python (BILSTM Experimentos)"
uv run --group notebooks jupyter lab
```

O primeiro comando cria `.venv` e instala as versões registradas em `uv.lock`. O segundo registra o ambiente como kernel do Jupyter. O terceiro abre o JupyterLab. Se o terminal estiver em outra pasta, navegue primeiro até a raiz, onde ficam `pyproject.toml` e `uv.lock`.

Para instalar apenas as dependências Python de análise, sem Jupyter:

```powershell
uv sync
```

Para adicionar uma dependência comum:

```powershell
uv add nome-do-pacote
```

Para adicionar uma ferramenta de notebook/desenvolvimento:

```powershell
uv add --group notebooks nome-do-pacote
```

Faça alterações de dependências com `uv add` para atualizar `pyproject.toml` e `uv.lock` juntos. Não edite versões travadas manualmente.

## 2. Fluxo de um experimento

Cada experimento deve ser compreensível no notebook e executável pelo pipeline:

1. Crie ou atualize um notebook em `notebooks/`, com objetivo, hipótese, dados usados e resultados esperados.
2. Salve os parâmetros em um arquivo próprio em `configs/`. Registre seed, split por vídeo/cena, dimensões de entrada e parâmetros do modelo. Não deixe uma configuração importante escondida numa célula.
3. Coloque pré-processamento, treino, avaliação e geração de métricas em comandos reutilizáveis de `scripts/`.
4. No notebook, chame esses comandos com os parâmetros da configuração e mostre o comando, o estado e os resultados. Use o notebook para análise e gráficos, não para reimplementar o pipeline.
5. Grave cada execução numa pasta identificada em `output/`, incluindo configuração efetiva, log, métricas e figuras. Guarde checkpoints ali ou num armazenamento de artefatos apropriado.
6. Documente a origem e licença dos dados, a unidade do split e a versão do código usada. Nunca misture vídeos/cenas entre treino, validação e teste se isso puder causar vazamento.

Forma de chamada que os scripts deverão seguir quando o pipeline for implementado:

```powershell
uv run python scripts/<comando>.py --config configs/<experimento>.toml --output output/<id-da-execucao>
```

Os comandos de pipeline ainda não existem neste repositório. Substitua os marcadores pelos nomes reais quando os scripts forem criados; não copie este exemplo como se já fosse executável.

## 3. Onde cada coisa fica

| Preciso guardar… | Use… | Exemplos |
| --- | --- | --- |
| Uma análise que apresenta parâmetros e gráficos | `notebooks/` | `01_metricas_base.ipynb`, `02_robustez.ipynb` |
| Um comando do pipeline | `scripts/` | preparação, treino, avaliação, robustez |
| Parâmetros de uma execução | `configs/` | `baseline.toml`, `robustez-brilho.toml` |
| Dataset original | `data/raw/` | vídeos e rótulos recebidos, sem alterações |
| Dados convertidos/splits | `data/processed/` | metadados ou recortes produzidos pelo pipeline |
| Métricas, logs, pesos e figuras geradas | `output/<id-da-execucao>/` | `metrics.csv`, `environment.json`, gráficos |
| Decisões, método e contexto do capítulo | `docs/` | protocolo, decisões e referências |

Dados e resultados em `data/` e `output/` são ignorados pelo Git. Versione as configurações e o código necessários para reproduzi-los, não cópias de vídeos, pesos grandes ou saídas geradas. Se uma figura final precisar ser versionada para o texto, copie apenas essa figura para `docs/` ou para o destino editorial acordado e registre como foi gerada.

## 4. Regras para manter o projeto navegável

- Procure primeiro na tabela acima; não crie outra pasta para uma categoria já coberta.
- Use nomes curtos, descritivos e numerados para notebooks, por exemplo `01_baseline.ipynb` e `02_validacao_externa.ipynb`.
- Um notebook corresponde a uma pergunta ou etapa científica. Não crie um notebook por tentativa de parâmetro; registre variações na configuração e na saída da execução.
- Mantenha implementação em `scripts/` e apresentação em `notebooks/`.
- Crie uma nova subpasta somente quando houver mais de um tipo de conteúdo recorrente que não caiba nos locais definidos; antes, atualize este guia com a justificativa.
- Não mova nem renomeie as pastas já existentes sem atualizar este guia e o README.

## 5. Escopo e dependências pendentes

O plano atual prioriza MobileNetV2 + BiLSTM com atenção, métricas expandidas, robustez e validação externa. O dataset, o código original, os pesos, os logs e a definição de framework ainda precisam ser recebidos/confirmados. Por isso, este repositório configura o ambiente de notebook e análise, mas ainda não declara comandos de treino nem dependências específicas de PyTorch ou TensorFlow.
