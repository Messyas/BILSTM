# Pipeline scripts

Os scripts sao entradas finas para o pipeline compartilhado em `src/bilstm_pipeline/`. O fluxo ativo usa apenas os comandos de pesquisa e o downloader:

| Script | Finalidade |
| --- | --- |
| `download_dataset.py` | Baixar a versao configurada do RLVS para `data/raw/`. |
| `research.py prepare` | Inventariar videos e criar manifesto com particoes. |
| `research.py train` | Treinar a reconstrucao proposta no servidor. |
| `research.py evaluate` | Avaliar teste reservado, robustez ou base externa. |

Consulte `uv run python scripts/research.py --help` e siga o [guia do servidor](../docs/plano-de-implementacao/EXECUCAO_NO_SERVIDOR.md). Foram removidos comandos genericos de avaliacao de CSV e divisao de manifestos; as etapas reais usam o pipeline de pesquisa.
