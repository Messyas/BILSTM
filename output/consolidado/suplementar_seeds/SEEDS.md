# Material suplementar — variação entre seeds de treino

Média ± desvio padrão amostral (mínimo–máximo) entre seeds de treino. Seeds: `densenet121`: 1, 2, 3, 4, 5, 42, `inception_v3`: 1, 2, 3, 4, 5, 42, `mobilenet_v2`: 1, 2, 3, 4, 5, 42. Os resultados principais do capítulo usam a seed 42 (protocolo fechado); esta tabela é um adendo.

**O que varia:** apenas `training.seed` (inicialização das camadas novas, ordem dos lotes, dropout e, na robustez, a semente do ruído gaussiano). **O que é fixo:** o split por vídeo (manifesto da seed 42), o conjunto de teste (402 vídeos), o RWF-2000 (2000 vídeos), a meta de FPR na validação e todo o protocolo. Para cada seed, o limiar foi recalibrado na validação e congelado antes do teste. A dispersão mede, portanto, a instabilidade do treino, não a variação amostral do teste, que é coberta pelos IC por bootstrap da tabela principal.

## Treino e limiar

| Métrica | densenet121 (n=6) | inception_v3 (n=6) | mobilenet_v2 (n=6) |
|---|---|---|---|
| Acurácia de validação (best.pt) | 0,952 ± 0,011 (0,940–0,969) | 0,925 ± 0,013 (0,906–0,940) | 0,955 ± 0,006 (0,950–0,962) |
| Limiar congelado | 0,473 ± 0,351 (0,114–0,974) | 0,887 ± 0,166 (0,587–0,998) | 0,728 ± 0,166 (0,469–0,986) |

## Teste no RLVS

| Métrica | densenet121 (n=6) | inception_v3 (n=6) | mobilenet_v2 (n=6) |
|---|---|---|---|
| Acurácia | 0,939 ± 0,004 (0,933–0,943) | 0,889 ± 0,025 (0,863–0,918) | 0,955 ± 0,008 (0,945–0,968) |
| Recall (Violence) | 0,946 ± 0,022 (0,915–0,965) | 0,823 ± 0,064 (0,756–0,900) | 0,951 ± 0,012 (0,940–0,965) |
| Precisão (Violence) | 0,933 ± 0,020 (0,915–0,968) | 0,949 ± 0,016 (0,933–0,975) | 0,959 ± 0,007 (0,950–0,970) |
| Especificidade | 0,931 ± 0,022 (0,910–0,970) | 0,955 ± 0,017 (0,935–0,980) | 0,959 ± 0,007 (0,950–0,970) |
| F1 macro | 0,939 ± 0,004 (0,933–0,943) | 0,888 ± 0,026 (0,862–0,918) | 0,955 ± 0,008 (0,945–0,968) |
| FPR | 0,069 ± 0,022 (0,030–0,090) | 0,045 ± 0,017 (0,020–0,065) | 0,041 ± 0,007 (0,030–0,050) |

## Validação externa no RWF-2000 (limiar congelado de cada seed)

| Métrica | densenet121 (n=6) | inception_v3 (n=6) | mobilenet_v2 (n=6) |
|---|---|---|---|
| Acurácia | 0,560 ± 0,016 (0,542–0,589) | 0,545 ± 0,014 (0,527–0,562) | 0,579 ± 0,015 (0,560–0,602) |
| Recall (Violence) | 0,588 ± 0,068 (0,452–0,637) | 0,536 ± 0,104 (0,361–0,661) | 0,620 ± 0,077 (0,556–0,749) |
| Precisão (Violence) | 0,559 ± 0,022 (0,537–0,587) | 0,547 ± 0,013 (0,522–0,560) | 0,574 ± 0,017 (0,550–0,591) |
| Especificidade | 0,531 ± 0,081 (0,476–0,682) | 0,554 ± 0,100 (0,394–0,699) | 0,538 ± 0,081 (0,387–0,615) |
| F1 macro | 0,558 ± 0,017 (0,540–0,588) | 0,541 ± 0,019 (0,516–0,561) | 0,576 ± 0,017 (0,553–0,600) |
| FPR | 0,469 ± 0,081 (0,318–0,524) | 0,446 ± 0,100 (0,301–0,606) | 0,462 ± 0,081 (0,385–0,613) |

## Robustez: Δ acurácia vs. limpo (teste RLVS)

| Métrica | densenet121 (n=6) | inception_v3 (n=6) | mobilenet_v2 (n=6) |
|---|---|---|---|
| Brilho | -0,005 ± 0,011 (-0,022–0,010) | -0,015 ± 0,016 (-0,037–0,005) | -0,012 ± 0,005 (-0,020–-0,005) |
| Contraste | -0,002 ± 0,012 (-0,017–0,012) | -0,013 ± 0,007 (-0,020–-0,002) | 0,004 ± 0,005 (0,000–0,012) |
| Ruído gaussiano | -0,060 ± 0,034 (-0,124–-0,035) | -0,080 ± 0,027 (-0,107–-0,037) | -0,125 ± 0,023 (-0,152–-0,090) |
| Oclusão central | -0,219 ± 0,078 (-0,313–-0,132) | -0,217 ± 0,059 (-0,308–-0,134) | -0,007 ± 0,007 (-0,020–0,000) |
| JPEG por quadro | -0,030 ± 0,015 (-0,052–-0,012) | -0,012 ± 0,013 (-0,035–0,002) | -0,022 ± 0,012 (-0,045–-0,007) |
| Recodificação H.264 | -0,008 ± 0,011 (-0,025–0,007) | 0,002 ± 0,005 (-0,005–0,007) | 0,003 ± 0,005 (-0,002–0,010) |

Valores por run em `por_run.csv`.
