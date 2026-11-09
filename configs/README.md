# Configuracoes de experimentos

Guarde aqui os arquivos TOML versionados com parametros cientificos reproduziveis, separados por protocolo ou experimento. O notebook pode apresentar substituicoes experimentais explicitas; a configuracao efetiva completa acompanha a execucao.

Nao coloque caminhos exclusivos de uma maquina, credenciais, dados, resultados, configuracao de bibliotecas ou estilo de graficos neste diretorio. `rlvs_baseline.toml` define os parametros alinhados aos notebooks originais enviados por Erick Farias (`FEATURE_EXTRACTION.ipynb` e `parametro_alterado_definitivo_densenet121_att(1).ipynb`):

- `[split]`: divisao `0.64 / 0.16 / 0.20` com `seed = 42` (correspondente a `test_size = 0.20` e `validation_split = 0.20` sobre os 80% de treino no codigo original, resultando em 1280 / 320 / 400 videos nos 2000 videos do RLVS).
- `[video]`: `frames = 15`, `width = 100`, `height = 100`, com amostragem por intervalo inteiro `total_frames // frames` e normalizacao em `[0, 1]`.
- `[model]`: escolha de `backbone` (`"mobilenet_v2"`, `"densenet121"` ou `"inception_v3"`), `pretrained = true`, `trainable_layers = 80`, `pooling` (`"avg"` para `GlobalAveragePooling2D` ou `"flatten"`), `hidden_size = 32` na BiLSTM, `attention = true` ou `false` (atencao por soma ponderada com 79 parametros para 15 quadros), `dense_units = [256, 128, 64, 32]` e `dropout = 0.25`.
- `[training]`: `optimizer = "sgd"`, `learning_rate = 0.01`, `momentum = 0.9`, `weight_decay = 0.0`, `epochs = 100`, `batch_size = 64` (ou `128` nas variacoes da Tabela 1 do artigo), `ReduceLROnPlateau` em `validation_loss` (`lr_factor = 0.6`, `lr_patience = 5`, `min_learning_rate = 0.00005` ou `0.0005`) e `EarlyStopping` em `validation_accuracy` (`early_stopping_patience = 10`).
