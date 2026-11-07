# Componentes disponíveis sem dataset

Os módulos abaixo estão implementados e podem ser usados independentemente do formato final dos vídeos. O notebook `notebooks/00_infrastructure_demo.ipynb` e a CLI `scripts/experiment.py` executam um exemplo completo **sintético**; as métricas desse exemplo não são resultados da pesquisa. O treino, o modelo original, a leitura de vídeos e o protocolo científico ainda precisam ser integrados quando os materiais estiverem disponíveis.

O código, os nomes de arquivos e o texto dos notebooks usam inglês; comentários e docstrings são escritos em inglês e português. Os exemplos abaixo são para quem desenvolve o pipeline: notebooks de pesquisa continuam usando apenas a API de alto nível.

## 1. Métricas e intervalos

`bilstm_pipeline.metrics.binary_metrics(y_true, y_pred)` recebe vetores binários de mesmo tamanho. A classe positiva é `1`; a matriz tem linhas verdadeiras e colunas previstas, na ordem `[[TN, FP], [FN, TP]]`. O retorno contém acurácia, precisão, recall e F1 por classe, suporte, F1 macro/ponderado, especificidade e FPR. Valor `None` indica divisão indefinida; o F1 macro é indefinido quando o F1 de alguma classe não é estimável. A média ponderada usa o suporte das classes.

```python
from bilstm_pipeline.metrics import binary_metrics, grouped_bootstrap

metrics = binary_metrics([0, 1, 0, 1], [0, 1, 1, 1])
interval = grouped_bootstrap(
    [0, 1, 0, 1], [0, 1, 1, 1], ["video-a", "video-a", "video-b", "video-b"],
    metric="accuracy", n_resamples=1000, seed=42,
)
```

`grouped_bootstrap` reamostra grupos inteiros e calcula o intervalo percentil. Para comparação pareada, informe `paired_pred` para o cenário B; o resultado será B menos A nos mesmos grupos reamostrados. O retorno informa quantidade de reamostragens válidas e descartadas quando a métrica é indefinida. Descartar muitas reamostragens pode tornar o intervalo pouco confiável e condicionado às amostras válidas; examine essa contagem antes de relatá-lo. O intervalo considera as observações fornecidas e o modelo fixo; não mede variabilidade de treinamento. A equipe precisa confirmar a unidade independente antes do uso científico.

## 2. Manifestos e divisão por grupos

`bilstm_pipeline.dataset` oferece `validate_records`, `group_split`, `validate_split` e `save_manifest`. Cada registro tem `sample_id`, `video_id`, `group_id` e `source` como strings; `label` pode ser 0, 1 ou ausente. Opcionalmente, `sha256` identifica duplicatas exatas conhecidas. A divisão usa grupos inteiros, é reproduzível pela seed e exige ao menos três grupos.

`group_split` distribui **grupos**, não a quantidade de vídeos ou janelas, e não faz estratificação. Assim, proporções e classes por partição podem diferir das metas e uma classe pode faltar. Confira o resumo produzido por `validate_split` e decida o protocolo antes de executar pesquisa. O validador detecta o mesmo grupo, vídeo ou hash conhecido em partições diferentes; ele não detecta automaticamente versões editadas de um mesmo vídeo nem garante que grupos independentes tenham sido definidos corretamente.

`save_manifest` exige nome, origem e licença e salva os registros, atribuições e contagens. Ele não copia arquivos de vídeo nem descobre permissões. Não inclua segredos ou dados pessoais no manifesto e mantenha os artefatos de dados em `data/` ou `output/`, ignorados pelo Git.

## 3. Interface do modelo e registro de resultados

`bilstm_pipeline.model.ModelAdapter` define `predict_scores(batch)`: o adaptador real deverá devolver uma probabilidade da classe positiva por amostra. `predict(adapter, batch, threshold=...)` valida os valores em `[0, 1]` e produz os rótulos. A interface não determina arquitetura, pré-processamento, treinamento ou framework; esses detalhes dependem do código original.

`bilstm_pipeline.results.save_evaluation` recebe IDs únicos, rótulos verdadeiros, predições, origem e unidade; opcionalmente recebe scores e limiar. Ele verifica a compatibilidade entre scores, limiar e predições e cria `evaluation.json` e `predictions.csv` em um diretório novo, recusando sobrescrita. A unidade pode ser janela, vídeo ou outra unidade efetivamente usada; uma predição por janela não equivale automaticamente a um alerta.

As execuções novas usam nomes de arquivo e chaves em inglês. `load_run` e as funções de apresentação continuam lendo as execuções anteriores em português, sem alterar seus arquivos.

## 4. Gráficos

`bilstm_pipeline.plots.plot_confusion_matrix` exporta a matriz em contagens com eixos explícitos; `plot_binary_metrics` exporta as métricas disponíveis da classe positiva. Ambos geram PDF e PNG em estilo centralizado com `matplotlib.rc_context`, mantendo os parâmetros globais intactos. O exemplo sintético vincula figuras, parâmetros, origem e unidade ao mesmo diretório de execução. Confirme o tamanho final, os formatos e as fontes aceitos pela editora antes da publicação.

## 5. Perturbações

`bilstm_pipeline.robustness` contém `brightness`, `contrast`, `gaussian_noise` e `occlusion`. A entrada é uma sequência `(tempo, altura, largura, canais)`, com um ou três canais: `uint8` de 0–255 ou ponto flutuante entre 0 e 1. A saída mantém forma e tipo, sem alterar a entrada; valores são limitados à faixa válida.

- Brilho e contraste usam o mesmo parâmetro em todos os quadros; contraste é centrado em 0,5.
- Ruído gaussiano usa uma seed e amostras independentes por pixel e quadro. Isso pode produzir variação temporal artificial; não demonstra comportamento de uma câmera real.
- Oclusão aplica o mesmo retângulo em todos os quadros. Ela não equivale a simular aglomeração.

O exemplo registra intensidade, seed e regiões em `component_demo.json`, junto das comparações pareadas sintéticas. As intensidades ali são ilustrativas; condições científicas, validade dos rótulos, níveis e unidades precisam ser decididos e documentados para a base real.

## Verificação

Na raiz do projeto, rode `uv run python -m unittest discover -s tests -v` para verificar casos numéricos conhecidos, divisão e vazamento por grupo, bootstrap pareado, perturbações, registro e leitura de execuções antigas. O notebook sintético também pode ser aberto pelo JupyterLab conforme o [guia de uso](../GUIA_DE_USO.md).
