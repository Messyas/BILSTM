# Arquitetura proposta

## Objetivo e estado

Organizar experimentos reproduzíveis sobre o modelo de detecção de conflitos com notebooks que expliquem a pergunta, apresentem os parâmetros experimentais e exibam os resultados. O processamento, a configuração do ambiente e a produção padronizada de figuras pertencem ao pipeline Python. O desenho a seguir é **planejado**: os módulos e as interfaces exemplificadas ainda não existem.

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
| `notebooks/` | Expor objetivo, identificador da configuração, eventuais parâmetros do experimento, execução e apresentação de tabelas/gráficos. | Existe; notebooks de pesquisa são futuros. |
| `configs/` | Arquivos TOML versionados com parâmetros científicos e regras de execução reproduzíveis. | Existe; o esquema está pendente. |
| `scripts/` | Entradas finas de terminal que interpretam argumentos, invocam a API e retornam código de saída. Sem duplicar o pipeline. | Existe; comandos são futuros. |
| `src/bilstm_pipeline/` | Pacote planejado: validação de parâmetros, orquestração, dados/splits, modelos, treino, avaliação, robustez, validação externa, artefatos e apresentação. | **Ainda não existe.** Criar módulos por responsabilidade e somente quando necessários. |
| `data/raw/`, `data/processed/` | Originais preservados e dados derivados, respectivamente. | Existem; conteúdo científico ainda depende de recebimento. |
| `output/<id>/` | Configuração efetiva, registro do ambiente, logs, métricas, predições, figuras e, se aplicável, pesos. | Diretório raiz existe; os artefatos são futuros e ignorados pelo Git. |
| `docs/` | Protocolo, decisões, fontes e andamento. | Existe. |

O pacote `src/` requer alterar `pyproject.toml`, que hoje define `package = false`: adotar um backend de build compatível com `uv`, instalar o projeto em modo editável via `uv sync` e documentar o import. Não depender de `sys.path` alterado em células nem de diretório de trabalho por acaso. Caminhos de configuração relativos à raiz do projeto, como no exemplo abaixo, devem ser resolvidos por uma função central tanto a partir da raiz quanto de `notebooks/`; caminhos absolutos são aceitos explicitamente e entradas ausentes produzem erro claro. Os dados locais podem ficar fora do repositório; caminhos são resolvidos por argumentos ou variáveis de ambiente documentadas, nunca com credenciais em TOML versionado.

### Contrato da API e das entradas

O pacote terá uma API pequena para notebooks e scripts, com as mesmas regras. Nomes a definir na implementação; responsabilidades previstas:

1. **Preparar**: ler e validar a configuração de experimento; aplicar substituições explícitas autorizadas; gerar identificador único e registrar o estado inicial.
2. **Executar**: orquestrar etapas solicitadas, falhar claramente quando faltar entrada e registrar sucesso ou falha; nunca substituir silenciosamente um resultado anterior.
3. **Apresentar**: ler artefatos e exibir parâmetros efetivos, resumo, tabelas e figuras por meio de funções de alto nível. Gráficos também podem ser produzidos via linha de comando.

Parâmetros de experimento incluem, conforme protocolo confirmado, seed, definição de split, checkpoint, tamanho de lote, taxa de aprendizado, limiar, janelas e níveis de perturbação. TOML versionado é a fonte padrão. Um notebook **pode declarar e passar somente substituições experimentais explícitas**, como `seed` ou intensidade de ruído; configurações de bibliotecas, ambiente, estilo ou montagem do pipeline não pertencem às células. A API valida os campos, rejeita valores desconhecidos e registra a configuração efetiva completa após substituições. A API deve deixar visível o que foi efetivamente executado.

Identificadores de execução devem ser únicos (por exemplo, timestamp em UTC com sufixo gerado), sem sobrescrever execuções anteriores; tentativa de reuso falha com mensagem clara, salvo quando for uma operação explícita de leitura. Versões, revisão Git disponível, hardware, framework, seeds, fontes dos dados e logs devem ser registrados sem segredos ou dados pessoais desnecessários. As referências a checkpoints e dados externos devem ser rastreáveis sem exigir que entrem no Git.

### Modelo de notebook

Cada notebook terá uma pergunta científica e, idealmente, uma única importação pública do projeto. Os nomes abaixo são **ilustrativos**, ainda não implementados:

```python
from bilstm_pipeline import notebook as experimento
```

```python
# Identificação e parâmetros científicos explícitos, quando necessários.
configuracao = "configs/baseline.toml"
parametros = {"seed": 42}
```

```python
# API planejada: solicita execução e devolve um identificador rastreável.
execucao = experimento.executar(configuracao, parametros=parametros)
experimento.mostrar_parametros(execucao)
```

```python
experimento.mostrar_resultados(execucao)
```

Essas células documentam a interface pretendida; não são comandos executáveis hoje. Se o usuário quiser apenas inspecionar um resultado anterior, a futura API deve oferecer operação de leitura por ID que não inicie treino novamente. O notebook pode incluir texto explicativo e conclusões, mas não `matplotlib`, `seaborn`, `pandas`, framework de treino, variáveis de ambiente, `sys.path`, `rcParams`, carregadores de dados ou funções de métricas. Qualquer nova biblioteca necessária à apresentação deve ser encapsulada na API do pacote. Evitar salvar nos notebooks cópias de dados sensíveis ou saídas volumosas; vincular as conclusões ao ID de execução.

### Configurações e saídas

- `configs/`: parâmetros do experimento e protocolo reproduzível em TOML. Esquema e exemplos reais só após verificar o código e os dados originais. Não armazenar credenciais, caminhos específicos de uma máquina, ajustes de bibliotecas ou `rcParams` ali.
- Configuração técnica e defaults de apresentação: código ou recursos versionados no pacote (`.mplstyle` quando conveniente). Configurações locais de ambiente ficam no ambiente do usuário e documentadas sem segredos.
- `output/<id>/`: prever `config_efetiva.toml`, `ambiente.json`, `manifesto.json`, `metricas/`, `figuras/`, logs e eventuais predições/checkpoints. Os nomes são um contrato proposto a ser finalizado durante a implementação; versionar pequenos metadados de protocolo em `docs/` quando necessário, sem versionar dados privados. Manter o identificador, a revisão, a fonte e a unidade de análise associados a cada figura e tabela.
- O esquema precisa contemplar execução concluída ou falha, e impedir que um artefato parcial seja apresentado como resultado concluído. Não registrar valor científico inventado como se viesse de uma execução real; usar somente dados sintéticos identificados nas verificações.

## Padrão de figuras para publicação

Implementar um módulo de visualização no pacote com Matplotlib e Seaborn quando útil. Centralizar estilo e exportação, e aplicá-los num contexto (`matplotlib.rc_context` ou estilo equivalente) para evitar efeitos globais e diferenças entre notebooks. As funções devem receber dados/resultados e devolver figuras ou caminhos; os notebooks só chamam a API de apresentação.

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

Comandos desejados, sujeitos à implementação e documentados como exemplos futuros:

```text
uv run python scripts/<etapa>.py --config configs/<experimento>.toml --output output/<id>
```

O comando e o notebook devem produzir o mesmo comportamento para os mesmos parâmetros, em especial esquema de validação, saída e regras de erro. A API será a fonte única da lógica. Nenhum comando de treino é oferecido como pronto enquanto o backlog correspondente não estiver concluído.
