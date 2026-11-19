# Continuação da GS de Applied Computer Vision - Calçada Segura

Este documento reconstrói o contexto a partir dos arquivos preservados, do enunciado original e do GitHub consultado em 19/11/2026. Não é uma transcrição integral de chats anteriores. Use o estado confirmado e os arquivos, sem presumir que uma promessa antiga foi executada.

## Pedido e objetivo

Concluir a entrega de Applied Computer Vision da FIAP, Global Solution CitizenTech 2026.2, usando treinamento real, avaliação documentada, código completo no GitHub e PDF pronto para a tarefa do Teams. O novo material enviado é `Damaged Sidewalks.v6i.yolov8(1).zip`, também enviado sem o sufixo `(1)`.

O produto Calçada Segura detecta defeitos no passeio. O componente de visão usa Python, Ultralytics YOLO11n e OpenCV, recebe webcam, stream ou MP4, desenha detecções, confirma ocorrências em quadros consecutivos e pode guardar foto e evento numa fila local para envio à API. O contrato original tem as classes `buraco`, `calcada_quebrada` e `obstaculo`. A integração demonstrada é CV -> API própria local -> RPA -> painel local. Não afirmar que isso já está implantado no MVP remoto de outro integrante sem conferir.

## Equipe

| Integrante | RM |
| --- | --- |
| Enzo Luiz Goulart | 99666 |
| João Pedro Cruz | 98650 |
| Tiago Paulino | 551169 |
| Victor Eid | 98668 |

## Requisitos originais de CV

Fonte: página 44 de `Desafio-FIAP-GS-CitizenTech 4ES.pdf`.

- Treinamento customizado de arquitetura compatível com o problema de detecção de objetos ou realidade aumentada.
- GitHub com documentação e código completo.
- Vídeo da solução funcionando.
- PDF com título, integrantes, resumo de exatamente 250 palavras, introdução, objetivo de exatamente 50 palavras, explicação técnica das ferramentas, metodologia passo a passo, exemplos de resultados e conclusão.
- Avaliação: documento até 5 pontos; vídeo até 5 pontos.

O enunciado recuperado registra entrega no Teams até 19/11/2026 às 23h55. Conferir essa informação no enunciado se for necessário orientar o envio. O envio ao Teams não foi realizado por este assistente.

## Arquivos preservados e acesso

- [Enunciado original](https://chatgpt.com/library?file_id=libfile_fe677b0cb1f08191979ead4f062e0db1&account_id=personal).
- [Pacote completo de CV anterior](https://chatgpt.com/library?file_id=libfile_cc6c43c673388191a217b23afa1c211d&account_id=personal): 126 arquivos, código, pesos, evidências, PDF e vídeos.
- [Modelo, dados e retreinos locais anteriores](https://chatgpt.com/library?file_id=libfile_f7d7bc243db0819196dab56a47c46bbe&account_id=personal).
- [Relatório dos ajustes anteriores](https://chatgpt.com/library?file_id=libfile_0353c4b9973c8191b86903b0bf1c22fc&account_id=personal).
- [Novo ZIP Roboflow](https://chatgpt.com/library?file_id=libfile_7e67bfc642b08191bf97dc561e1303ea&account_id=personal).
- Repositório: https://github.com/enzoolu/calcada-segura-cv .

Os links de arquivos pessoais exigem a conta autorizada; não substituem os links públicos que o avaliador precisa abrir no GitHub.

## Baseline e limitações anteriores

O detector principal é YOLO11n, não YOLOv8. A palavra `yolov8` no nome do ZIP identifica o formato de exportação das anotações; não é um checkpoint treinado e não obriga trocar a arquitetura.

As evidências anteriores registram mAP50 de 36,65% no teste público de 112 imagens, média de três classes. No campo, o detector marcou piso tátil danificado em 7 de 9 quadros selecionados. Em 16 quadros históricos de São Paulo, com 7 cavidades anotadas, não localizou nenhuma dessas caixas de buraco. Não confundir número de quadros com número de defeitos físicos distintos.

No pacote de ajustes locais há 44 imagens anotadas, 46 caixas, duas classes com exemplos positivos e 4 imagens ambíguas excluídas. Dois retreinos foram executados em CPU. O primeiro reduziu falsos alarmes, mas diminuiu os acertos de calçada quebrada de 11/17 para 3/17 nos mesmos 27 quadros reservados. O especialista em buracos também não melhorou a localização das quatro vistas de uma cavidade reservada. Por isso o modelo original continuou recomendado. As anotações locais foram feitas por inspeção visual pelo assistente e não têm validação independente de especialista.

Os vídeos existentes são demonstrações de inferência real sobre gravações, renderizadas offline. A taxa de reprodução do MP4 não comprova inferência em tempo real. Há também código de câmera, que pode ser executado no computador do usuário. Não alegar acesso à webcam neste ambiente.

## Auditoria do novo dataset

ZIP inspecionado e extraído com sucesso. SHA256: `ef85704ae09f7252ac7a354bf028016148a688f1046b13387bc56feffe36df46`.

- Origem: https://universe.roboflow.com/damagedsidewalks/damaged-sidewalks/dataset/6 .
- Licença informada no pacote: CC BY 4.0; preservar atribuição e os dois READMEs originais.
- 389 imagens JPEG, todas com 640 x 640 pixels.
- Uma classe original: `Losa_Agrietada`, correspondente ao escopo de calçada trincada/quebrada.
- Não há classes anotadas de buraco ou obstáculo neste ZIP.
- Partições originais: 278 treino, 75 validação, 36 teste.
- 613 caixas no total. Diversas imagens são negativos com arquivo de rótulo vazio; isso é válido.
- Nenhum rótulo inválido ou arquivo de imagem duplicado por SHA256 foi encontrado na auditoria.

Foram encontrados quatro grupos de recortes de uma mesma imagem divididos entre partições: 3228, 3242, 3468 e 2435. Para impedir essa sobreposição, todos os recortes de cada grupo foram levados à partição mais reservada, com precedência teste > validação > treino. Nenhuma imagem originalmente de teste foi transferida para treino.

| Partição corrigida | Imagens | Caixas | Negativos |
| --- | ---: | ---: | ---: |
| Treino | 275 | 422 | 167 |
| Validação | 76 | 118 | 45 |
| Teste | 38 | 73 | 22 |

Esse agrupamento por quadro de origem não comprova independência por rua, rota ou gravação. Essa limitação deve constar no relatório.

## Novo treinamento executado nesta continuação

Foi criado `visao/src/retreinar_roboflow.py`, com etapas separadas de preparação, treino e avaliação. Ele preserva os pesos originais e registra auditoria, manifesto e protocolo em `visao/evidencias/roboflow-v6/`.

O novo especialista usa apenas `calcada_quebrada`, com ID 0 no conjunto novo. O baseline continua com `calcada_quebrada` no ID 1. Para comparar, é necessário mapear essas saídas à mesma classe; não comparar IDs sem remapeamento e não misturar o mAP de uma classe com a média histórica de três classes.

Configuração: pesos iniciais `visao/evidencias/treino-publico/best.pt`, YOLO11n, até 30 épocas, 416 px, batch 8, CPU, AdamW, taxa inicial 0,001, primeiras 10 camadas congeladas, seed 42, aumento moderado, mosaic desativado e patience 10. O treino terminou e gerou `visao/evidencias/roboflow-v6/best.pt`; o registro possui 29 épocas por parada antecipada. O melhor checkpoint foi escolhido pela validação. Isso comprova execução do treino, não melhora do modelo.

A avaliação comparativa foi concluída: no teste corrigido de 38 imagens, mAP50 de 0,12% no original e 3,73% no especialista; este teve 9 acertos, 64 falsos alarmes e 64 omissões no limiar 0,10. O desempenho continua insuficiente para promover o especialista. O modelo original de três classes permanece padrão. Consulte `comparacao.json` para as métricas completas.

## Publicação e bloqueios confirmados

Na última conferência da branch `main`, o GitHub ainda tinha somente o README, commit `582ee5a488d5f3fbca85c22931b92fd9b5541021`. A conexão GitHub tem permissão de escrita. O envio direto com Git local falhou por falta de autenticação nesse transporte; isso não demonstra um limite de tamanho do GitHub. Tentativas alternativas de preparar os binários foram interrompidas sem confirmação. Não afirmar que houve commit, push ou publicação de pesos sem checar a árvore pública e o novo SHA.

## Prompt de execução para outro chat

Continue a entrega de Applied Computer Vision do Calçada Segura a partir deste contexto e dos arquivos vinculados. Leia os requisitos originais e os arquivos atuais; preserve tudo o que já foi executado. Confira o estado atual do GitHub antes de escrever. Use o ZIP Roboflow v6 de uma classe para treinamento real e preserve o modelo original de três classes. Se os novos pesos e resultados desta continuação estiverem disponíveis, reutilize-os e não reinicie o treino sem necessidade.

Conclua a comparação original versus especialista nas mesmas imagens, com remapeamento correto das classes, mesmo tamanho de entrada e mesmo critério de IoU. Escolha checkpoint e limiar apenas na validação e congele-os antes de avaliar o teste. Relate mAP50, mAP50-95, precisão, recall, F1, falsos positivos e omissões. Separe avaliação pública e diagnóstico nos vídeos locais. Só recomende a troca quando os resultados a sustentarem; um treinamento que não melhora ainda deve ser relatado honestamente.

Finalize o código e a documentação, execute verificações apropriadas, produza vídeo com predições reais e PDF cumprindo todas as seções e contagens de palavras. Publique os arquivos descompactados, pesos, vídeo e PDF no repositório de CV; não publique apenas um ZIP nem só o README. Verifique o commit e o acesso público aos links. Prepare o PDF final para o usuário enviar na tarefa do Teams. Não informe envio ao Teams sem que ele tenha ocorrido.

Informe de forma precisa qual etapa foi concluída e qual bloqueou. Durante o trabalho, inclua uma estimativa em minutos em cada atualização, distinguindo tempo de treino e tempo restante total. Não invente métricas, evidências, commits, links públicos ou aprovações.

## Forma final de entrega acordada

O usuário fará a publicação no próprio computador. Preparar PDF e ZIP completo; não insistir em envio remoto pelo assistente. Extrair o conteúdo na pasta clonada, fazer git add, commit e push, conferir os links públicos e então enviar o PDF no Teams. Não houve confirmação de novo commit remoto nesta continuação.

## Pacote final produzido

PDF final de 6 páginas, com resumo de 250 e objetivo de 50 palavras, links do repositório e vídeo, métricas medidas e limitações. O ZIP reúne 1.040 arquivos; todos os arquivos individuais têm menos de 100 MB. O vídeo principal tem 87,95 segundos. Foram aprovados oito testes existentes e verificações básicas das métricas. A publicação e o envio ao Teams serão realizados pelo usuário.
