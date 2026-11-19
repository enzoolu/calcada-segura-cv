# Calçada Segura — anotação, retreino e demonstração

Trabalho concluído em 19/11/2026 com as filmagens enviadas nesta conversa.

## Resultado principal

**As duas tentativas de retreino não demonstraram melhora nas cenas reservadas. O modelo original permanece recomendado como padrão.** O primeiro ajuste diminuiu falsos alarmes, mas perdeu muitos acertos de calçada quebrada. O especialista em buracos teve bom resultado na pequena validação, mas não acertou a localização das quatro vistas da cavidade reservada.

Foram concluídos:

- 44 imagens revisadas e anotadas no formato YOLO: 38 positivas e 6 negativas.
- 46 caixas de referência no total, cobrindo `buraco` e `calcada_quebrada`.
- 4 imagens ambíguas separadas, sem participação no treino nem na avaliação.
- Duas rodadas de treino executadas em CPU, partindo de pesos já existentes do projeto.
- Comparações com parâmetros e resultados salvos, além dos pesos originais e experimentais.
- Demonstração offline com detecções reais do modelo original e vídeo de comparação com o especialista experimental.

As caixas de referência foram desenhadas por inspeção visual pelo assistente multimodal e revisadas em pranchas de sobreposição. Não têm validação independente por especialista. Os arquivos de referência e suas imagens de conferência estão disponíveis para revisão.

## Anotações

As coordenadas normalizadas foram definidas a partir dos quadros orientados. Os positivos receberam caixas para as principais regiões visíveis de dano; os negativos revisados receberam arquivos de rótulo vazios. O material candidato a remendo e a obstáculo de papelão permaneceu fora. Microfissuras isoladas não foram anotadas exaustivamente.

Mapa de classes preservado no conjunto principal:

| ID | Classe |
| --- | --- |
| 0 | buraco |
| 1 | calcada_quebrada |
| 2 | obstaculo |

Não há positivos validados de obstáculo neste lote. A classe continua no contrato do detector original, mas sua capacidade de reconhecer obstáculos não pôde ser reavaliada com estes dados.

## Rodada 1 — ajuste do detector de três classes

A separação inicial foi conservadora por família visual e gravação. Todas as gravações com piso tátil, inclusive o vídeo 8372 e suas cópias comprimidas, ficaram fora do treino. Como as localizações não foram fornecidas, não afirmamos que as partições representam ruas ou rotas independentes.

| Partição | Imagens | Caixas de buraco | Caixas de calçada quebrada |
| --- | ---: | ---: | ---: |
| Treino | 14 | 3 | 13 |
| Validação | 3 | 0 | 3 |
| Avaliação reservada | 27 | 10 | 17 |

Treino: 30 épocas, 640 px, batch 4, AdamW, taxa inicial 0,0003 e primeiras 10 camadas congeladas, seed 42. O checkpoint foi escolhido pela validação; as anotações reservadas foram concluídas e revisadas antes de inspecionar as predições nessa partição.

A comparação principal usa 384 px, IoU de acerto de pelo menos 0,50 e os limiares operacionais do projeto: 0,10 para buraco, 0,30 para calçada quebrada e 0,50 para obstáculo. Também foi feita uma checagem secundária a 640 px.

| Métrica nas mesmas 27 imagens, a 384 px | Original | Ajustado |
| --- | ---: | ---: |
| Acertos de calçada quebrada | 11 de 17 | 3 de 17 |
| Falsos alarmes de calçada quebrada | 9 | 0 |
| F1 de calçada quebrada | 59,46% | 30,00% |
| Acertos de buraco | 0 de 10 | 0 de 10 |
| mAP50 médio das duas classes com referência | 25,21% | 25,13% |

O ajuste deixou o detector mais conservador no limiar usado pelo aplicativo. A redução de falsos alarmes veio acompanhada de perda grande de sensibilidade; não justifica substituir os pesos atuais. A validação é especialmente pequena e contém apenas a classe de calçada quebrada.

Arquivos: `avaliacao/comparacao.json`, `avaliacao/por-imagem.csv`, `avaliacao/protocolo-congelado.json` e os resultados completos em `treino/ajuste-local/`.

## Rodada 2 — especialista em buracos

Após examinar a rodada 1, foi realizada uma nova rodada de desenvolvimento, reorganizada por gravação para incluir mais exemplos de buraco no treino. Esta rodada é exploratória e **não constitui confirmação em teste externo cego**, pois o primeiro diagnóstico já havia sido observado.

O especialista parte de `evidencias/refino-buraco/best.pt`, já existente no projeto. Treinou apenas a classe buraco, com as outras regiões servindo como fundo para essa classe. O modelo original continua responsável por calçada quebrada e obstáculo na combinação experimental.

| Partição da rodada 2 | Imagens | Caixas de buraco |
| --- | ---: | ---: |
| Treino | 28 | 6 |
| Validação | 6 | 3 |
| Avaliação de desenvolvimento | 10 | 4 |

Validação: gravação 8363. Avaliação de desenvolvimento: gravações 8360 e 8372 inteiras fora do treino, com todas as vistas selecionadas mantidas juntas. O vídeo 8372 não foi usado no treino de nenhuma rodada.

Treino: 60 épocas, 640 px, batch 4, AdamW, taxa inicial 0,001, primeiras 10 camadas congeladas, seed 42. O limiar foi escolhido somente na validação, antes das predições desta rodada na partição de avaliação. O resultado selecionado foi confiança 0,40: 2 acertos, nenhum falso alarme e 1 falta nas 3 vistas de uma única abertura de validação.

| Buracos nas mesmas 10 imagens de desenvolvimento | Original | Original + especialista |
| --- | ---: | ---: |
| Acertos com IoU ≥ 0,50 | 0 de 4 | 0 de 4 |
| Falsos alarmes | 0 | 1 |
| Buracos não localizados | 4 | 4 |

O original também foi verificado a 640 px e continuou sem acertos dessas quatro vistas. A combinação preservou exatamente as detecções do modelo original para calçada quebrada e obstáculo, pois só troca a saída de buraco.

O especialista não foi aprovado para substituir a saída padrão de buraco. Os resultados e o limiar estão em `especialista/comparacao.json`, `especialista/perfil-hibrido.json` e `especialista/protocolo-rodada-2.json`.

As quatro vistas positivas de desenvolvimento representam **uma única cavidade física**, não quatro buracos distintos. Não somar as métricas das duas rodadas nem comparar diretamente as taxas entre partições diferentes.

## Demonstrações entregues

- `demonstracao-calcada-segura.mp4`: vídeo 8372 processado com o modelo original preservado. Mostra caixas, classes e confiança reais, sem inserir as caixas de referência da anotação. É uma demonstração do detector atual, não de uma melhora do retreino.
- `comparacao-experimental.mp4`: a mesma sequência, lado a lado, com o original e a combinação com especialista. Mostra o comportamento real, inclusive falhas.
- `demo-deteccoes.json`: todas as detecções emitidas por quadro, com confiança e coordenadas normalizadas.

As demonstrações são renderizações offline, sem transmissão de ocorrências para a API/RPA e sem áudio ambiente. A sequência foi mantida, com redução de resolução e conversão para 30 fps. A taxa de reprodução do arquivo não comprova inferência em tempo real.

A filmagem já enviada serve para demonstrar o funcionamento atual; não é necessário gravar outro vídeo só para apresentação. O detector pode marcar o pavimento trincado e ainda deixar de reconhecer a cavidade como buraco. Essa limitação permanece e não foi escondida no vídeo.

## Por que o material ainda não bastou para melhorar

O lote tem poucas cenas físicas e muitas vistas correlacionadas. Na rodada 2, as seis caixas de buraco do treino correspondem principalmente a duas cenas: uma abertura parcialmente ocupada por resíduos e outra entre pedras. A cavidade de avaliação, aberta em cimento junto ao piso tátil, tem aparência diferente. Esse é um diagnóstico visual compatível com os erros observados, não uma causa comprovada por experimento isolado.

O próximo material útil é de outros buracos em cimento, com variação de tamanho, distância, iluminação e presença de folhas/resíduos, junto com pisos íntegros semelhantes. Uma gravação de outro local deve ser reservada antes do treino seguinte. Mais quadros do mesmo ponto não substituem essa variedade.

## Reprodução

O pacote inclui as imagens, rótulos, manifestos, sobreposições para revisão, scripts, configurações, logs e pesos. As dependências executadas foram Python 3.12.14, PyTorch 2.15.0+cpu e Ultralytics 8.4.212. `requirements-exatos.txt` registra o ambiente completo.

Na pasta extraída, em ambiente Python 3.12 com FFmpeg disponível:

```bash
python -m pip install -r requirements.txt
python scripts/treinar.py
python scripts/comparar.py
python scripts/treinar_especialista.py
python scripts/avaliar_especialista.py
python scripts/demonstrar.py --video caminho/1000268372.mp4
```

Os YAMLs do pacote são relativos ao diretório em que estão armazenados. Os scripts resolvem os caminhos a partir da própria localização. Para repetir as demonstrações, fornecer o vídeo 8372 original enviado nesta conversa; ele não é duplicado no pacote.

`modelos/modelo-original.pt` é o padrão recomendado. `modelo-ajustado.pt` e `especialista-ajustado.pt` são pesos experimentais não aprovados nesta comparação. Não trocar o padrão do aplicativo por eles com base apenas no treino ou na validação. Os pesos originais do repositório permanecem intactos.
