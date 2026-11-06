# Orientações para agentes e colaboradores

## Antes de alterar o projeto

1. Leia o [README](README.md), o [guia de uso](docs/GUIA_DE_USO.md), a [arquitetura planejada](docs/plano-de-implementacao/ARQUITETURA.md) e o [backlog](docs/plano-de-implementacao/BACKLOG.md). Para decisões científicas, consulte também a [avaliação crítica](docs/avaliacao-de-tudo/AVALIACAO_CRITICA.md).
2. Consulte `git status --short` e preserve alterações preexistentes. Escolha uma tarefa do backlog e confira suas dependências; não trate uma tarefa bloqueada como concluída.
3. Mantenha notebooks como interface: apenas uma importação da API de alto nível, identificação do experimento, parâmetros específicos quando necessário, chamadas e apresentação de resultados. Configuração de ambiente, caminhos locais, estilo de gráficos, lógica de processamento e imports de bibliotecas do pipeline ficam fora dos notebooks.
4. Coloque parâmetros reproduzíveis em `configs/` e implementações compartilhadas no pacote planejado `src/bilstm_pipeline/`; use `scripts/` como entradas finas para a linha de comando. Não invente dados, métricas ou resultados científicos.
5. Crie pastas somente quando cumprirem uma responsabilidade prevista ou houver necessidade concreta documentada. Atualize os guias quando mudar contratos, comandos ou organização.

## Conclusão e commits

- Execute as verificações adequadas à alteração e registre evidências relevantes no backlog. Só marque `[x]` quando os critérios de aceite estiverem atendidos; explique bloqueios com `[ ]` e mantenha a tarefa aberta.
- Ao concluir uma unidade coerente, atualize o backlog no mesmo conjunto de alterações e faça um commit local pequeno e descritivo, incluindo apenas os arquivos próprios daquela unidade. Exemplos: `docs: definir protocolo de avaliação` ou `feat: adicionar leitura de parâmetros`.
- Antes de adicionar arquivos ao índice, examine `git status` e o diff. Não use `git add .` quando puder incluir trabalho alheio, dados, segredos ou artefatos gerados; selecione os arquivos explicitamente. Preserve alterações e arquivos preexistentes, inclusive já adicionados ao índice.
- Verifique o conteúdo staged antes do commit. Se arquivos alheios já estiverem staged, não os inclua nem altere o índice deles; faça o commit com isolamento seguro ou registre o bloqueio e informe ao usuário. Se Git ou identidade de autoria não estiver configurado, informe o motivo e prossiga com o trabalho possível. Não faça push, rebase, reset ou alterações destrutivas sem solicitação.
- O registro do progresso deve poder ser revisado via backlog e `git log`; um hash não precisa ser inserido no próprio commit que ele identifica.
