# Configuracoes de experimentos

Guarde aqui os arquivos TOML versionados com parametros cientificos reproduziveis, separados por protocolo ou experimento. O notebook pode apresentar substituicoes experimentais explicitas; a configuracao efetiva completa acompanha a execucao.

Nao coloque caminhos exclusivos de uma maquina, credenciais, dados, resultados, configuracao de bibliotecas ou estilo de graficos neste diretorio. `rlvs_baseline.toml` define os parametros propostos para treino e avaliacao do RLVS; eles ainda precisam ser comparados ao protocolo original.

`weight_decay` explicita a regularizacao L2 usada pelo Adam; `0.0` mantem o comportamento padrao anterior do otimizador.
