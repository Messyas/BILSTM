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

## 2. Baixar o dataset

O pesquisador identificou o [Real Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) como a base usada. A versão 1 é a versão fixada neste projeto; a correspondência com a versão e a divisão do experimento anterior ainda precisa ser confirmada.

1. Crie uma conta no Kaggle e consulte a página do dataset para verificar o acesso, os termos e as condições de uso dos arquivos.
2. Em [Kaggle Settings → API](https://www.kaggle.com/settings/api), gere um token. A [documentação oficial do Kaggle](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication) também descreve login pelo navegador e credenciais antigas `kaggle.json`.
3. No PowerShell, configure o token somente na sessão atual, sem colocá-lo no repositório:

   ```powershell
   $env:KAGGLE_API_TOKEN = "SEU_TOKEN_DO_KAGGLE"
   ```

   No macOS/Linux, use `export KAGGLE_API_TOKEN="SEU_TOKEN_DO_KAGGLE"`. Outra opção é seguir a documentação do Kaggle para `~/.kaggle/access_token` ou `~/.kaggle/kaggle.json`.
4. Na raiz do projeto, execute:

   ```powershell
   uv sync --all-groups
   uv run --group data python scripts/download_dataset.py
   ```

Os arquivos serão extraídos em `data/raw/rlvs-v1/`, ignorados pelo Git. A API pública do Kaggle informava cerca de 1,95 GB para os arquivos em 07/11/2026; é necessário espaço adicional para o ZIP e a extração. O arquivo `download_metadata.json` registra referência, versão, data e SHA-256 do ZIP para rastreabilidade. O ZIP é temporário e é removido após a extração.

O comando não sobrescreve um destino existente. Para baixar o dataset externo recomendado para avaliação *cross-dataset* (**RWF-2000**, `vulamnguyen/rwf2000`, com 2.000 vídeos de câmeras de vigilância reais em `Fight` e `NonFight`) ou outras bases do Kaggle (`scvd`, `hockey`):

```powershell
uv run --group data python scripts/download_dataset.py --dataset rwf2000
```

Uma falha de autenticação, rede ou extração encerra o comando sem publicar a pasta final. Após o download, siga o [guia de execução no servidor](plano-de-implementacao/EXECUCAO_NO_SERVIDOR.md) para inventariar os vídeos, treinar e avaliar a reconstrução proposta.

## 3. Executar o fluxo de pesquisa

O fluxo usa o RLVS e uma reconstrucao proposta do modelo. Siga o [guia de execucao no servidor](plano-de-implementacao/EXECUCAO_NO_SERVIDOR.md) para validar GPU, preparar os dados, treinar e avaliar.

Na raiz, prepare o manifesto:

```powershell
uv run --group training python scripts/research.py prepare --config configs/rlvs_baseline.toml
```

Depois, no servidor de treinamento:

```powershell
uv run --group training python scripts/research.py train --config configs/rlvs_baseline.toml
uv run --group training python scripts/research.py evaluate --run-id ID_DO_TREINO --robustness
```

As etapas tambem estao nos notebooks `01_prepare_dataset.ipynb`, `02_train_baseline.ipynb` e `03_evaluate_test.ipynb`. Os comandos mostram o ID do treino e o caminho da avaliacao em `output/`. O conjunto de teste fica reservado ate a avaliacao; nao use seus resultados para escolher parametros.

Para dividir por cena ou origem, prepare um CSV completo `sample_id,group_id` e passe `--groups caminho/grupos.csv` ao comando `prepare`. Use o mesmo ID para videos relacionados. Sem esse mapeamento, o padrao agrupa apenas hashes identicos. Consulte os limites no guia do servidor.

A avaliacao externa e opcional e depende de uma base distinta, licenciada e com rotulos compativeis. Prepare configuracao e manifesto proprios; use `04_evaluate_external.ipynb` ou o comando documentado no guia do servidor.

Os notebooks sao interfaces finas: identificam a configuracao, chamam a API do projeto e apresentam artefatos. Parametros reproduziveis ficam em `configs/`; processamento, metricas e graficos ficam no pacote Python.

## 4. Onde cada coisa fica

| Preciso guardar… | Use… | Exemplos |
| --- | --- | --- |
| Uma análise que apresenta parâmetros e gráficos | `notebooks/` | `01_prepare_dataset.ipynb`, `03_evaluate_test.ipynb` |
| Logica compartilhada de processamento e apresentacao | `src/bilstm_pipeline/` | pacote instalado pelo uv; pipeline de pesquisa compartilhado |
| Uma entrada de terminal que chama a API | `scripts/` | preparação, treino, avaliação, robustez |
| Parâmetros de uma execução | `configs/` | `baseline.toml`, `robustness-brightness.toml` |
| Dataset original | `data/raw/` | vídeos e rótulos recebidos, sem alterações |
| Dados convertidos/splits | `data/processed/` | metadados ou recortes produzidos pelo pipeline |
| Métricas, logs, pesos e figuras geradas | `output/<id-da-execucao>/` | `metrics.csv`, `environment.json`, gráficos |
| Decisões, método e contexto do capítulo | `docs/` | protocolo, decisões e referências |

Dados e resultados em `data/` e `output/` são ignorados pelo Git. Versione as configurações e o código necessários para reproduzi-los, não cópias de vídeos, pesos grandes ou saídas geradas. Se uma figura final precisar ser versionada para o texto, copie apenas essa figura para `docs/` ou para o destino editorial acordado e registre como foi gerada.

## 5. Regras para manter o projeto navegável

- Procure primeiro na tabela acima; não crie outra pasta para uma categoria já coberta.
- Use nomes curtos, descritivos, em inglês e numerados para notebooks, por exemplo `01_baseline.ipynb` e `02_external_validation.ipynb`.
- Um notebook corresponde a uma pergunta ou etapa científica. Não crie um notebook por tentativa de parâmetro; registre variações na configuração e na saída da execução.
- Mantenha implementação e estilo de gráficos no pacote; use `scripts/` como entradas de terminal e `notebooks/` como interface de alto nível.
- Consulte o [backlog](plano-de-implementacao/BACKLOG.md) antes de iniciar trabalho. Atualize tarefas e evidências ao concluir, e siga [AGENTS.md](../AGENTS.md) para commits locais pequenos e rastreáveis.
- Crie uma nova subpasta somente quando houver mais de um tipo de conteúdo recorrente que não caiba nos locais definidos; antes, atualize este guia com a justificativa.
- Não mova nem renomeie as pastas já existentes sem atualizar este guia e o README.

## 6. Escopo e dependências pendentes

O plano atual prioriza MobileNetV2 + BiLSTM com atenção (permitindo alternar para DenseNet121 ou InceptionV3 e ligar/desligar a atenção), métricas expandidas, robustez e validação externa. A base RLVS e os notebooks originais de Erick Farias foram recebidos e seus parâmetros foram incorporados a `configs/rlvs_baseline.toml` e ao pacote Python; os arquivos de pesos `.keras` originais e o mapeamento de cenas independentes ainda precisam ser confirmados.

A [arquitetura](plano-de-implementacao/ARQUITETURA.md) detalha a API e os padrões de figuras. O [guia de componentes](plano-de-implementacao/COMPONENTES.md) descreve as funções reutilizáveis e seus limites; as etapas científicas seguem no backlog.
