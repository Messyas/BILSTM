# Configurações de experimentos

Guarde aqui uma configuração versionada por experimento ou protocolo, usando TOML. Registre os parâmetros necessários para repetir a execução (incluindo seed, split por vídeo/cena, entrada do modelo e hiperparâmetros). Parâmetros experimentais declarados em notebooks serão substituições explícitas validadas pela futura API; a configuração efetiva completa ficará junto dos resultados. Não guarde caminhos locais, credenciais, dados, resultados, configuração de bibliotecas ou estilo de gráficos neste diretório.

O formato e o esquema de parâmetros serão definidos junto com a implementação do pipeline, depois que o código e o dataset originais forem recebidos. Não crie arquivos de configuração fictícios antes disso.
