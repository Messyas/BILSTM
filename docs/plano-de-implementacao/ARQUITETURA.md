# Arquitetura proposta

## Objetivo e estado

Organizar experimentos reproduzíveis sobre o modelo de detecção de conflitos com notebooks que expliquem a pergunta, apresentem os parâmetros experimentais e exibam os resultados. O processamento, a configuração do ambiente e a produção padronizada de figuras pertencem ao pipeline Python. O pacote e a interface já oferecem uma **demonstração sintética**; as etapas científicas continuam planejadas.

## Responsabilidades e dependências

```text
notebooks/  ─── chama API de alto nível ──┐
                                         ▼
scripts/ ────── chama a mesma API ──► src/bilstm_pipeline/
                                         │  lê parâmetros
                                         ├──► configs/
                                         │  lê dados e pesos autorizados
                                         ├──► data/
                                         │  grava artefatos por execução
                                         └──► output/<id>/
```

| Local | Responsabilidade | Estado |
| --- | --- | --- |
| `notebooks/` | Expor objetivo, identificador da configuração, eventuais parâmetros do experimento, execução e apresentação de tabelas/gráficos. | Demonstração sintética disponível; notebooks de pesquisa são futuros. |
| `configs/` | Arquivos TOML versionados com parâmetros científicos e regras de execução reproduzíveis. | Esquema sintético disponível; esquema científico pendente. |
| `scripts/` | Entradas finas de terminal que interpretam argumentos, invocam a API e retornam código de saída. Sem duplicar o pipeline. | `experimento.py` executa ou consulta a demonstração. |
| `src/bilstm_pipeline/` | Pacote de validação, orquestração, artefatos e apresentação; dados/splits, modelos, treino e avaliação são planejados. | Instalável via uv; funciona para demonstração sintética. |
| `data/raw/`, `data/processed/` | Originais preservados e dados derivados, respectivamente. | Existem; conteúdo científico ainda depende de recebimento. |
| `output/<id>/` | Configuração efetiva, registro do ambiente, logs, métricas, predições, figuras e, se aplicável, pesos. | Diretório raiz existe; os artefatos são futuros e ignorados pelo Git. |
| `docs/` | Protocolo, decisões, fontes e andamento. | Existe. |

O pacote `src/` usa Hatchling e é instalado em modo editável via `uv sync`. Não depende de `sys.path` alterado nas células. Caminhos relativos à raiz do projeto, como no exemplo abaixo, são resolvidos tanto a partir da raiz quanto de `notebooks/`; caminhos absolutos são aceitos e entradas ausentes produzem erro. A forma de informar dados locais externos será definida no pipeline científico, sem credenciais em TOML versionado.

### Contrato da API e das entradas

O pacote oferece uma API pequena para notebooks e scripts, com as mesmas regras. Responsabilidades:

1. **Preparar**: ler e validar a configuração de experimento; aplicar substituições explícitas autorizadas; gerar identificador único e registrar o estado inicial.
2. **Executar**: orquestrar etapas solicitadas, falhar claramente quando faltar entrada e registrar sucesso ou falha; nunca substituir silenciosamente um resultado anterior.
3. **Apresentar**: ler artefatos e exibir parâmetros efetivos, resumo, tabelas e figuras por meio de funções de alto nível. Gráficos também podem ser produzidos via linha de comando.

Parâmetros de experimento incluem, conforme protocolo confirmado, seed, definição de split, checkpoint, tamanho de lote, taxa de aprendizado, limiar, janelas e níveis de perturbação. TOML versionado é a fonte padrão. Um notebook **pode declarar e passar somente substituições experimentais explícitas**, como `seed` ou intensidade de ruído; configurações de bibliotecas, ambiente, estilo ou montagem do pipeline não pertencem às células. A API valida os campos, rejeita valores desconhecidos e registra a configuração efetiva completa após substituições. A API deve deixar visível o que foi efetivamente executado.

Na demonstração, identificadores incluem timestamp em UTC e sufixo aleatório; o reuso é recusado. São registrados versões Python e de bibliotecas, sistema, arquitetura da máquina, revisão Git disponível, seed, origem sintética e log. Na implementação científica, ampliar o registro para hardware de processamento, framework, fontes reais e checkpoints, sem segredos ou dados pessoais desnecessários.

### Modelo de notebook

Cada notebook terá uma pergunta científica e, idealmente, uma única importação pública do projeto. O exemplo abaixo é executável para demonstrar a infraestrutura com dados sintéticos:

```python
from bilstm_pipeline import notebook as experimento
```

```python
# Identificação e parâmetros científicos explícitos, quando necessários.
configuracao = "configs/demonstracao_sintetica.toml"
parametros = {"seed": 42}
```

```python
# Solicita a demonstração sintética e devolve um identificador rastreável.
execucao = experimento.executar(configuracao, parametros=parametros)
experimento.mostrar_parametros(execucao)
```

```python
experimento.mostrar_resultados(execucao)
```

As funções `mostrar_parametros(id)` e `mostrar_resultados(id)` consultam uma execução concluída sem iniciá-la de novo. A demonstração não executa o modelo científico. O notebook pode incluir texto explicativo e conclusões, mas não `matplotlib`, `seaborn`, `pandas`, framework de treino, variáveis de ambiente, `sys.path`, `rcParams`, carregadores de dados ou funções de métricas. Qualquer nova biblioteca necessária à apresentação deve ser encapsulada na API do pacote. Evitar salvar nos notebooks cópias de dados sensíveis ou saídas volumosas; vincular as conclusões ao ID de execução.

### Configurações e saídas

- `configs/`: parâmetros do experimento e protocolo reproduzível em TOML. Há um exemplo sintético; esquema e exemplos científicos dependem de verificar o código e os dados originais. Não armazenar credenciais, caminhos específicos de uma máquina, ajustes de bibliotecas ou `rcParams` ali.
- Configuração técnica e defaults de apresentação: código ou recursos versionados no pacote (`.mplstyle` quando conveniente). Configurações locais de ambiente ficam no ambiente do usuário e documentadas sem segredos.
- `output/<id>/`: a demonstração grava `config_efetiva.toml`, `ambiente.json`, `manifesto.json`, `execucao.log`, `dados_sinteticos.csv` e figuras PDF/PNG. Métricas, predições e checkpoints permanecem planejados. Manter o identificador, a revisão, a fonte e a unidade de análise associados a cada figura e tabela.
- O esquema precisa contemplar execução concluída ou falha, e impedir que um artefato parcial seja apresentado como resultado concluído. Não registrar valor científico inventado como se viesse de uma execução real; usar somente dados sintéticos identificados nas verificações.

## Padrão de figuras para publicação

O módulo `graficos.py` aplica o estilo e exporta a figura da demonstração usando Matplotlib e `matplotlib.rc_context`. As visualizações científicas e suas regras específicas ainda precisam ser implementadas; Seaborn pode ser usado quando útil. Os notebooks só chamam a API de apresentação.

**Padrão inicial a adaptar às normas do veículo:**

- Usar tamanho final físico da figura, com fonte serifada disponível e fallback; base 10 pt, rótulos 11 pt, título até 12 pt, ticks e legenda 9 pt. Verificar legibilidade na dimensão final; não presumir a presença de uma fonte específica.
- Como ponto de partida, coluna de 8 cm = **3,15 polegadas** de largura (`8 / 2,54`), por exemplo `figsize=(8/2.54, 5.5/2.54)`; largura de 17 cm = **6,69 polegadas**, por exemplo `(17/2.54, 9/2.54)`. São exemplos, não dimensões universais; confirmar coluna e proporção com a publicação. A documentação do [Matplotlib explica as unidades de figura](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_size_units.html).
- Exportar PDF ou SVG vetorial quando aceito e PNG a 300 DPI, com opção de 600 DPI se exigido. DPI afeta imagens raster e elementos rasterizados; não aumenta a resolução de traços vetoriais puros. Checar fontes embutidas, transparências e formato conforme a editora; EPS depende das exigências e pode ter limitações de transparência.
- Usar layout que preserve legendas e rótulos; `bbox_inches="tight"` pode alterar o tamanho final exportado, então verificar as dimensões após exportação quando elas forem exigidas. Consulte [`savefig`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html).
- Rótulos com unidades e nomes de classes inequívocos; legendas que não cubram os dados; grade discreta quando ajudar; remover bordas superior/direita apenas quando apropriado ao gráfico. Reportar incerteza e unidade de análise quando necessárias.
- Usar paleta acessível adequada ao dado: por exemplo, `colorblind` para categorias e `viridis`/`cividis` para escalas ordenadas; complementar cores com marcadores, linhas ou texturas. Conferir legibilidade em escala de cinza. Veja as [orientações de paletas do Seaborn](https://seaborn.pydata.org/tutorial/color_palettes.html).
- Para matrizes de confusão, indicar explicitamente classe verdadeira e prevista, ordem das classes e se a matriz apresenta contagens ou proporções. Para comparações de robustez, indicar referência limpa, unidade e definição das barras de incerteza.

O estilo sugerido pelo usuário (`font.family`, tamanhos, grade discreta, `savefig.dpi`) é uma referência para o módulo central; a configuração não deve ser colada em notebooks. A [documentação do Matplotlib](https://matplotlib.org/stable/users/explain/customizing.html) descreve `rcParams` e folhas de estilo. Garantir que a aplicação do estilo não sobrescreva uma configuração global de outro código em execução.

## Interfaces de linha de comando

Comandos disponíveis para a demonstração, executados na raiz:

```text
uv run python scripts/experimento.py executar --config configs/demonstracao_sintetica.toml
uv run python scripts/experimento.py mostrar --id ID_DA_EXECUCAO
```

O comando e o notebook chamam a mesma API. A definição de comandos científicos permanece no backlog; nenhum comando de treino está pronto.
