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
2. Salve os parâmetros padrão em um arquivo próprio em `configs/`. Registre seed, split por vídeo/cena, dimensões de entrada e parâmetros do modelo conforme o protocolo. O notebook poderá declarar apenas substituições experimentais explícitas, registradas junto da configuração efetiva.
3. Coloque pré-processamento, treino, avaliação, métricas e apresentação no pacote Python em `src/bilstm_pipeline/`. Faça os comandos de `scripts/` chamarem a mesma API.
4. No notebook, use a API de alto nível do projeto para executar ou carregar uma execução e apresentar parâmetros efetivos, estado, tabelas e gráficos. Evite importar bibliotecas científicas diretamente ou configurar ambiente, gráficos e caminhos locais nas células.
5. Grave cada execução numa pasta identificada em `output/`, incluindo configuração efetiva, log, métricas e figuras. Guarde checkpoints ali ou num armazenamento de artefatos apropriado.
6. Documente a origem e licença dos dados, a unidade do split e a versão do código usada. Nunca misture vídeos/cenas entre treino, validação e teste se isso puder causar vazamento.

Para experimentar a infraestrutura com dados sintéticos, rode na raiz:

```powershell
uv run python scripts/experiment.py run --config configs/synthetic_demo.toml --seed 42 --points 12
```

O comando devolve o ID da execução. Consulte-a sem executar novamente, substituindo o exemplo pelo ID informado:

```powershell
uv run python scripts/experiment.py show --id ID_DA_EXECUCAO
```

`seed` e `points` são substituições opcionais dos valores do TOML. A configuração completa e efetiva fica em `output/<id>/effective_config.toml`; o manifesto identifica os dados como sintéticos. IDs já existentes não são sobrescritos. O exemplo também gera manifesto de dados, métricas, previsões, intervalos, perturbações e figuras sintéticas. Treino e avaliação de vídeos reais ainda não foram implementados.

Os nomes do script, da configuração e das funções do notebook foram alterados para inglês; comandos e imports antigos em português devem ser atualizados conforme os exemplos acima. As execuções anteriormente salvas em `output/` continuam disponíveis para consulta pelo novo comando `show` e pela API `load_run`.

Para reavaliar o CSV gerado pela demonstração, substitua `ID_DA_EXECUCAO` pelo ID recebido:

```powershell
uv run python scripts/evaluate.py --input output/ID_DA_EXECUCAO/evaluation/predictions.csv --output output/ID_DA_EXECUCAO/reevaluation --source synthetic --unit sample --threshold 0.5
uv run python scripts/split_dataset.py --input output/ID_DA_EXECUCAO/dataset_manifest.json --output output/ID_DA_EXECUCAO/new_split.json --seed 42
```

Esses scripts também aceitam arquivos próprios no mesmo formato; informe corretamente origem, unidade e limiar. Uma divisão por grupos não estratificada deve ser revisada antes do uso científico. As saídas precisam ser caminhos novos e seus diretórios pais devem existir.

## 3. Onde cada coisa fica

| Preciso guardar… | Use… | Exemplos |
| --- | --- | --- |
| Uma análise que apresenta parâmetros e gráficos | `notebooks/` | `01_baseline_metrics.ipynb`, `02_robustness.ipynb` |
| Lógica compartilhada de processamento e apresentação | `src/bilstm_pipeline/` | pacote instalado pelo uv; demonstração sintética disponível |
| Uma entrada de terminal que chama a API | `scripts/` | preparação, treino, avaliação, robustez |
| Parâmetros de uma execução | `configs/` | `baseline.toml`, `robustness-brightness.toml` |
| Dataset original | `data/raw/` | vídeos e rótulos recebidos, sem alterações |
| Dados convertidos/splits | `data/processed/` | metadados ou recortes produzidos pelo pipeline |
| Métricas, logs, pesos e figuras geradas | `output/<id-da-execucao>/` | `metrics.csv`, `environment.json`, gráficos |
| Decisões, método e contexto do capítulo | `docs/` | protocolo, decisões e referências |

Dados e resultados em `data/` e `output/` são ignorados pelo Git. Versione as configurações e o código necessários para reproduzi-los, não cópias de vídeos, pesos grandes ou saídas geradas. Se uma figura final precisar ser versionada para o texto, copie apenas essa figura para `docs/` ou para o destino editorial acordado e registre como foi gerada.

## 4. Regras para manter o projeto navegável

- Procure primeiro na tabela acima; não crie outra pasta para uma categoria já coberta.
- Use nomes curtos, descritivos, em inglês e numerados para notebooks, por exemplo `01_baseline.ipynb` e `02_external_validation.ipynb`.
- Um notebook corresponde a uma pergunta ou etapa científica. Não crie um notebook por tentativa de parâmetro; registre variações na configuração e na saída da execução.
- Mantenha implementação e estilo de gráficos no pacote; use `scripts/` como entradas de terminal e `notebooks/` como interface de alto nível.
- Consulte o [backlog](plano-de-implementacao/BACKLOG.md) antes de iniciar trabalho. Atualize tarefas e evidências ao concluir, e siga [AGENTS.md](../AGENTS.md) para commits locais pequenos e rastreáveis.
- Crie uma nova subpasta somente quando houver mais de um tipo de conteúdo recorrente que não caiba nos locais definidos; antes, atualize este guia com a justificativa.
- Não mova nem renomeie as pastas já existentes sem atualizar este guia e o README.

## 5. Escopo e dependências pendentes

O plano atual prioriza MobileNetV2 + BiLSTM com atenção, métricas expandidas, robustez e validação externa. O dataset, o código original, os pesos, os logs e a definição de framework ainda precisam ser recebidos/confirmados. Por isso, este repositório configura o ambiente de notebook e análise, mas ainda não declara comandos de treino nem dependências específicas de PyTorch ou TensorFlow.

A [arquitetura](plano-de-implementacao/ARQUITETURA.md) detalha a API e os padrões de figuras. O [guia de componentes](plano-de-implementacao/COMPONENTES.md) descreve as funções reutilizáveis e seus limites; as etapas científicas seguem no backlog.
