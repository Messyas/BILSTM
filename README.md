# Experimentos de detecção de conflitos em vídeo

Este repositório organiza os experimentos computacionais de um capítulo sobre **prontidão da infraestrutura de dados e governança territorial**. O caso empírico é a detecção de conflitos humanos em vídeo com uma CNN + BiLSTM; o foco técnico atual é aprofundar a avaliação do modelo MobileNetV2 + BiLSTM, com atenção, e produzir resultados reproduzíveis que apoiem a discussão do capítulo.

Os notebooks são a interface de experimentação: apresentam objetivo, parâmetros, execução e gráficos. O processamento fica em scripts Python reutilizáveis, chamados pelos notebooks, para manter a implementação fora das células e permitir executar o mesmo pipeline sem abrir o notebook. O dataset, o código original e os artefatos de execução ainda precisam ser recebidos; portanto, os experimentos não estão prontos para rodar.

## Começar

Instale o [uv](https://docs.astral.sh/uv/getting-started/installation/) e, na raiz do repositório, execute:

```powershell
uv sync --all-groups
uv run --group notebooks python -m ipykernel install --user --name bilstm-experimentos --display-name "Python (BILSTM Experimentos)"
uv run --group notebooks jupyter lab
```

No Jupyter, selecione o kernel **Python (BILSTM Experimentos)** e abra um notebook em `notebooks/`. Para Windows PowerShell, veja o [tutorial de uso](docs/GUIA_DE_USO.md); ele também descreve o fluxo de trabalho, a convenção para novos experimentos e onde cada tipo de arquivo deve ficar.

## Mapa do projeto

| Caminho | O que deve conter |
| --- | --- |
| `notebooks/` | Interfaces dos experimentos: motivação, parâmetros, chamadas ao pipeline, tabelas e gráficos. |
| `scripts/` | Comandos de pipeline independentes e reutilizáveis, sem lógica de apresentação de notebook. |
| `configs/` | Configurações versionadas dos experimentos, uma por execução identificável. |
| `data/raw/` | Dados originais, mantidos fora do Git. Não editar os arquivos de origem. |
| `data/processed/` | Dados derivados e divisões de treino, validação e teste, também fora do Git. |
| `output/` | Resultados gerados, logs, checkpoints e figuras; por padrão, fora do Git. |
| `docs/` | Contexto da pesquisa, decisões, documentação e tutoriais. |

Consulte `docs/GUIA_DE_USO.md` antes de criar arquivos ou pastas. Ele define o fluxo notebook → configuração → scripts → resultados e orienta a manter cada coisa no lugar previsto.

## Escopo científico atual

O plano de revisão prioriza métricas por classe e agregadas, matriz de confusão, taxa de falso alerta, intervalos de confiança, latência e throughput; testes de robustez; e validação externa (*cross-dataset*). A documentação também registra as exigências editoriais sobre originalidade, governança territorial, metodologia, salvaguardas éticas e autoria. O capítulo deve separar resultados efetivamente medidos de propostas futuras e tratar o sistema como apoio à decisão sob supervisão humana.

O dataset é um bloqueio registrado no contexto atual: origem, licença, distribuição de classes, anotação, split e artefatos precisam ser confirmados antes de definir ou executar o pipeline. A acurácia de 96,5% mencionada em documentos de contexto não é tratada aqui como resultado de teste confirmado.

## Ambiente

O `pyproject.toml` descreve as dependências comuns de análise e notebook; `uv.lock` fixa as versões resolvidas para reproduzir esse ambiente. O framework de deep learning será adicionado quando o código e os pesos originais forem recebidos e a compatibilidade com o servidor/GPU for conhecida.
