# Calçada Segura · Applied Computer Vision

**FIAP · Engenharia de Software 4ES · GS CitizenTech 2026.2**  
Enzo Luiz Goulart · RM 99666, João Pedro Cruz · RM 98650, Victor Eid · RM 98668 e Tiago Paulino · RM 551169.

O componente usa YOLO11n para detectar `buraco`, `calcada_quebrada` e `obstaculo`.
O modelo de três classes, os limiares e o manifesto da avaliação pública estão em
`evidencias/treino-publico/`. O modelo específico refinado para dano na calçada
está em `evidencias/refino-pune/`. Os pesos originais continuam disponíveis em
`evidencias/best.pt` para reproduzir a prova local que aparece no vídeo de RPA.

## Inferência

No Windows, o script `visao/Testar-Camera.ps1` na raiz configura o ambiente,
abre a câmera em tempo real e grava o vídeo em `saida/`. A opção `-Fonte 1`
escolhe outra câmera. A opção `-Pesos` aponta para um modelo retreinado.

O [vídeo narrado](evidencias/demonstracao-visao.mp4) mostra o detector no recorte
de campo, exemplos de avaliação e a configuração de obstáculos. O
[relatório técnico](docs/relatorio-visao.pdf) inclui os resultados por amostra.

Na pasta `visao`:

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -m src.inferir --pesos evidencias/treino-publico/best.pt --perfil evidencias/treino-publico/perfil.json --imgsz 384 --fonte 0 --device cpu
py -3.12 -m src.inferir --pesos evidencias/best.pt --fonte rota.mp4 --sem-janela --saida saida/deteccoes.mp4
```

Use `--fonte 0` para a câmera integrada, `--fonte 1` para uma segunda câmera ou a URL do stream do celular. Encerre com **Q**. O MP4 anotado contém as detecções e o terminal registra a taxa média de processamento daquela execução. Para reavaliar a classe com outro limiar, ajuste `--confianca` usando a validação. Mostre o limiar e informe o desempenho sobre o conjunto reservado.

Metadados podem ser enviados para a API. A opção abaixo usa a coordenada fixa de um ponto de teste. Ao demonstrar um vídeo gravado, explique que a posição foi informada pelo operador e o horário é o do processamento.

```powershell
$env:CALCADA_API_KEY = 'cole-a-chave-do-ambiente'
py -3.12 -m src.inferir --pesos evidencias/best.pt --fonte 0 --api http://127.0.0.1:8000 --lat -23.5614 --lon -46.6559
```

A inferência confirma a mesma classe em três quadros consecutivos com sobreposição entre caixas (IoU ≥ 0,30), limita novos registros por classe a cada 20 s e guarda o UUID em fila antes do POST. A API e o RPA completam a deduplicação. O cliente envia tipo, confiança e coordenada. Para colocar uma foto na apresentação, selecione um quadro revisado e borre manualmente rostos e placas antes de compartilhar.

## Reproduzir o experimento

- Fonte, informações de coleta e citação: [Indian Footpath Damage Segmentation Dataset, v2](https://zenodo.org/records/17154852), de Priti Chakurkar e colaboradores. O arquivo de licença declara CC BY 4.0. As máscaras binárias identificam danos na calçada.
- `evidencias/pune-imagens.json` lista os nomes dos pares usados. Extraia esses pares do ZIP original e mantenha as pastas `Train`, `Val` e `Test`. O conversor reduz o lado maior a até 960 pixels e extrai caixas por componente conectado, descartando áreas abaixo de 42 pixels² e lados abaixo de 5 pixels.
- Os splits de origem foram preservados. A escolha de amostras levou em conta o tamanho de cada arquivo e ocorreu antes do treino. Por isso, o teste mede desempenho na amostra, com evidência limitada sobre a diversidade do conjunto completo. O arquivo de pesos inicial foi o `yolo11n.pt` distribuído pelo Ultralytics. Referência da arquitetura e licença dos pesos em [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolo11.md).

```powershell
py -3.12 -m src.preparar_pune caminho/para/pares dataset/pune
py -3.12 -m src.treinar --dados dataset/pune/data.yaml --device cpu --epocas 28 --imgsz 480 --batch 12 --workers 0 --saida runs/pune
```

## Experimentos com bases públicas

- **Três classes.** O manifesto em `evidencias/treino-publico/manifesto.csv` contém origem, split, arquivo, SHA-256 e rótulo normalizado. Para buracos, usamos a base de vias de [Roma e Sacrofano](https://zenodo.org/records/17834373) (CC BY 4.0), com grupos por data no teste e na validação. Para obstáculos, usamos o [ROD-Dataset](https://huggingface.co/datasets/Abtinzandi/Obstacle-Detection-Dataset-YOLO/tree/103b38ccb6c80be269e5c7288e63b828d6f54e0b), fixado pelo commit, preservando os splits e usando quatro categorias de objetos: lixeira, cone, barreira cilíndrica e vaso. As fontes de terceiros mantêm suas licenças. As amostras originais dessa base não são redistribuídas no pacote. Para calçada quebrada, usamos as caixas derivadas das máscaras de Pune.
- **Refinamento em Pune.** A mesma avaliação em 30 imagens e 44 caixas obteve mAP50 de 10,51% com os pesos anteriores e 15,40% com os refinados. Precisão passou de 17,50% para 33,87% e recall caiu de 20,45% para 18,18%. Métricas da rodada anterior, incluindo 10,85% com outra configuração de avaliação, permanecem registradas em `comparacao-amostra-ampliada.json`.
- **Modelo especializado em buracos.** Nos mesmos 42 quadros de vias italianas, a 480 px, o mAP50 da classe foi de 13,50% com o modelo de três classes e 17,88% com o modelo específico em `evidencias/refino-buraco/best.pt`. Os dois números usam somente os quadros italianos. Para testar o especialista, informe `-Pesos visao/evidencias/refino-buraco/best.pt` no script de câmera. O desempenho ainda é baixo e a base retrata vias de veículos.
- **Limite.** Cada classe vem principalmente de uma fonte diferente. Buracos de vias e objetos anotados em calçada são aproximações para o problema local. O modelo de obstáculos identifica objetos candidatos. Uma inspeção contextual deve confirmar se o objeto bloqueia a passagem.

Para reproduzir o treino de três classes, organize os originais conforme o caminho
em `imagem` do manifesto. Em Pune, use os JPEGs processados pelo conversor. Nas
vias de Roma, extraia a entrada do ZIP indicada em `origem`. Para o ROD, use o
arquivo do commit informado e o split do manifesto (a pasta de validação na
fonte se chama `valid`).

```powershell
py -3.12 -m src.preparar_publico evidencias/treino-publico/manifesto.csv caminho/para/bases dataset/publico
py -3.12 -m src.experimentos tres-classes --dados dataset/publico/data.yaml --pesos yolo11n.pt --saida runs/publico
py -3.12 -m src.calibrar --pesos runs/publico/weights/best.pt --dados dataset/publico --saida runs/publico/perfil.json
```

### Reforço de buracos com imagens de celular

Adicionamos ao treino 41 fotos (35 positivas, 6 negativas, 110 caixas) da base de
[Giordani et al.](https://doi.org/10.5281/zenodo.17834373), CC BY 4.0. O script
exclui dias reservados, hashes SHA-256 repetidos e imagens muito parecidas pelo
dHash. Treinamos por 12 épocas, a 448 px, com AdamW, semente 42 e lote 12.
Os dois pesos foram avaliados a 384 px nos mesmos 112 arquivos do teste original.

| Métrica no teste reservado | Modelo padrão | Retreino |
| --- | ---: | ---: |
| AP50 de buraco | 10,70% | 12,67% |
| AP50 de calçada quebrada | 29,24% | 22,24% |
| AP50 de obstáculo | 70,01% | 67,06% |
| mAP50 | 36,65% | 33,99% |
| mAP50–95 | 21,77% | 22,96% |
| Buracos localizados de 44 (confiança ≥ 0,10, IoU ≥ 0,50) | 7 | 17 |
| Falsos alarmes de buraco no mesmo limiar | 31 | 68 |

O script de câmera mantém o modelo padrão para as três classes. Em sete fotos
independentes de passeios do Wikimedia Commons, os dois pesos perderam as quatro
cavidades anotadas no limiar de 0,10. São imagens escolhidas para uma checagem
exploratória. O manifesto dessa amostra guarda os links, hashes e caixas manuais.

O diretório `evidencias/reforco-roma/` inclui os pesos experimentais, perfil,
manifesto de treino, métricas, curvas, quadros anotados e demonstração offline.
O vídeo usa três imagens reservadas de vias italianas montadas como entrada
de vídeo. O aplicativo produz as caixas ao processar o arquivo.

Na pasta `visao`, com o conjunto público preparado e o ZIP de Roma intacto:

```powershell
py -3.12 -m src.refinar_buracos --base dataset/publico --roma caminho/para/pothole_dataset.zip --manifesto evidencias/treino-publico/manifesto.csv --dados-ampliados dataset/reforco-roma --pesos evidencias/treino-publico/best.pt --saida runs/reforco-roma
py -3.12 -m src.calibrar --pesos runs/reforco-roma/weights/best.pt --dados dataset/publico --imgsz 384 --saida runs/reforco-roma/perfil.json
py -3.12 -m src.inferir --pesos evidencias/reforco-roma/best.pt --perfil evidencias/reforco-roma/perfil.json --imgsz 384 --device cpu --fonte 0
```

### Ajustes sem novas fotos

A opção `-ReforcarBuracos` usa o retreino de Roma a 480 px e confiança 0,12
só para a classe buraco. As outras classes usam os pesos e os limiares
do modelo padrão. O modo executa duas inferências por quadro. A opção é
experimental e permanece desligada por padrão.

```powershell
# Na raiz do projeto
powershell -ExecutionPolicy Bypass -File .\visao\Testar-Camera.ps1 -ReforcarBuracos

# Na pasta visao, com o conjunto público original já preparado
py -3.12 -m src.comparar_buracos --dados dataset/publico --saida runs/comparacao-inferencia
```

No teste quadro a quadro, o ajuste recuperou 16 de 44 buracos em vias
e gerou 36 falsos alarmes. O padrão recuperou 9 e gerou 37 falsos alarmes.
Os dois omitiram as quatro cavidades anotadas na amostra independente
de sete fotos de calçadas. As contagens diferem da comparação em lote
porque o processamento quadro a quadro usa o redimensionamento da câmera.

O experimento de composições usa recortes dos polígonos de Roma sobre pisos de
Pune, com borda suavizada. Os dois componentes são exclusivamente de treino.
Cada cena composta preserva as caixas existentes e recebe uma caixa de buraco
sem sobreposição com elas. A validação e o teste permanecem idênticos.
Com 100 cenas compostas, o modelo detectou 13 de 44 buracos e produziu
38 falsos alarmes no teste de vias. Nas sete fotos de calçadas, produziu
quatro falsos alarmes e localizou zero das quatro cavidades anotadas.
Os pesos dessa rodada ficam disponíveis para reprodução do experimento.

```powershell
py -3.12 -m src.simular_cavidades --base dataset/reforco-roma --roma caminho/para/pothole_dataset.zip --manifesto evidencias/reforco-roma/manifesto-treino.json --saida dataset/cavidades-simuladas --pesos evidencias/reforco-roma/best.pt --treino runs/cavidades-simuladas
py -3.12 -m src.comparar_buracos --dados dataset/publico --pesos-simulado runs/cavidades-simuladas/weights/best.pt --saida runs/comparacao-simulado
```

As contagens, os parâmetros e o manifesto das composições ficam em
`evidencias/ajuste-buracos/`.

### Revisão de cavidades em calçadas de Pune

Anotamos 15 cavidades reais em imagens de treino de Pune e repetimos as
15 imagens revisadas para dar mais peso à classe. As partições originais
de validação e teste permaneceram byte a byte iguais. Uma checagem dedicada
de Pune usa sete fotos de validação e oito reservadas, revisadas manualmente.
A checagem anterior do Wikimedia Commons também foi usada como teste.

O novo treino acertou uma das três cavidades da validação pré-selecionada.
Nas duas checagens reservadas de calçada, o resultado foi de zero acertos
em oito cavidades. No teste original, ambos os retreinos detectaram 16
buracos a 480 px e confiança 0,15. O retreino de Roma teve 26 falsos
alarmes e a revisão de Pune teve 30. O README registra 0,12 como o limiar
padrão de `-ReforcarBuracos`, definido na validação original.

Para escolher um limiar mais estrito durante a demonstração:

```powershell
powershell -ExecutionPolicy Bypass -File .\visao\Testar-Camera.ps1 -ReforcarBuracos -ConfiancaBuracos 0.15
```

Os pesos experimentais e os dados da comparação ficam em
`evidencias/revisao-pune/`. Na pasta `visao`, com os conjuntos já preparados:

```powershell
py -3.12 -m src.revisar_cavidades --base dataset/reforco-roma --manifesto evidencias/revisao-pune/anotacoes.json --treino dataset/revisao-pune --checagem dataset/checagem-pune --pesos evidencias/reforco-roma/best.pt --ajuste runs/revisao-pune
py -3.12 -m src.avaliar_cavidades --dados dataset/publico --checagem dataset/checagem-pune --pesos runs/revisao-pune/weights/best.pt --saida runs/revisao-pune/comparacao.json
```

### Experimento com cavidades e gravações de São Paulo

A comparação com anotação manual usou 108 fotos de Pune, 42 quadros viários
reservados de Roma e 16 quadros de quatro gravações históricas de São Paulo.
Os novos retreinos localizaram zero das sete cavidades anotadas no teste
paulistano. O padrão segue como detector do script de câmera. Os pesos,
comandos e resultados por quadro estão em
[evidencias/cavidades-calcada](evidencias/cavidades-calcada/).

## Amostragem futura em São Paulo

1. Filme a partir da posição de uso da câmera. Reúna cavidades no passeio, placas quebradas, obstáculos reais, trechos livres e condições variadas de iluminação. Mantenha gravações de locais diferentes para o treino e o teste.
2. Separe os vídeos em `rotas/treino`, `rotas/validacao` e `rotas/teste`.
   Rode `python -m src.extrair_quadros rotas quadros --intervalo 2` na pasta `visao`
   para gerar o inventário e as imagens organizadas. Rotule com caixas justas. `buraco` é uma cavidade no piso. `calcada_quebrada` cobre dano superficial ou placa partida. `obstaculo` cobre um objeto que ocupa a faixa de circulação. Preserve as sequências de gravação inteiras em um único split. Reserve uma rota para o teste final e evoluções de treino para outras rotas.
   Se quiser acelerar a anotação, rode `python -m src.sugerir_rotulos quadros sugestoes --pesos evidencias/treino-publico/best.pt`. O CSV aponta as caixas a revisar. Copie para o conjunto somente os rótulos revisados e anote os quadros da rota de teste diretamente.
3. Use uma exportação YOLO no formato do Roboflow (pastas `train/valid/test` com `images/` e `labels/`). Para negativos revisados, inclua arquivo de rótulo vazio. Se complementar com uma fonte pública, registre versão, licença e classes. O dataset [Damaged Sidewalks](https://universe.roboflow.com/damagedsidewalks/damaged-sidewalks/dataset/1) contém `Bache` e rótulos de dano, úteis como candidatos ao treino. Revise o mapa de classes e exemplos antes de mesclar. Para preservar um teste local, mantenha imagens externas no treino.
4. Converta e treine no Colab com GPU. Use o mesmo teste reservado ao comparar modelos.

```powershell
py -3.12 -m src.preparar_yolo exportacao dataset/sp --mapa docs/classes.example.json
py -3.12 -m src.treinar --dados dataset/sp/data.yaml --epocas 60 --imgsz 640 --device 0 --saida runs/rota-sp
```

Para exportação já com as classes do contrato, edite o JSON de mapeamento para associar cada nome a si mesmo. Registre contagens de imagens e caixas por classe, mAP50, mAP50–95, precisão, recall e matriz de confusão. Guarde dois acertos e dois erros que ilustrem condições presentes no teste.

A recorrência de três quadros foi verificada em três testes de regressão
(`python -m unittest discover -s tests -v`). Em uma execução local, o modelo
de três classes enviou um registro de calçada quebrada e o RPA o classificou na
Sé com coordenada de teste. Evidência em
`evidencias/treino-publico/prova-api-rpa.json`.

## Gravação

A prova de campo usa um recorte gravado com celular. O detector confirmou um dano no piso tátil como `calcada_quebrada`. A opção `--anexar-foto` associa um JPEG do quadro à ocorrência confirmada e guarda ambos na fila até o envio. O timestamp registra a execução e a coordenada é informada pelo operador.

O recorte, a demo com as detecções e o resultado da triagem estão em `integracao/evidencias/campo/`, a partir da raiz. A demonstração usa um ponto de teste na Sé.

### Checagem de campo por quadro

Foram revisadas duas cenas do mesmo vídeo. O modelo principal marcou `calcada_quebrada` em 7 dos 9 quadros com o piso tátil danificado e marcou `obstaculo` indevidamente em 6 dos 8 quadros com os calçados do operador. A checagem mede presença por quadro em cenas selecionadas. O CSV da extração pode orientar a escolha de quadros em um novo vídeo.

Execute na raiz para reproduzir com os quadros e a referência incluídos:

```powershell
py -3.12 -m visao.src.avaliar_campo --quadros visao/evidencias/avaliacao-campo/quadros --referencia visao/evidencias/avaliacao-campo/referencia.json --pesos visao/evidencias/treino-publico/best.pt --perfil visao/evidencias/treino-publico/perfil.json --saida saida/avaliacao-campo.json
```

Para repetir com a faixa superior de 75%, acrescente `--obstaculo-ate-y 0.75`
e altere o nome da saída. As marcações indevidas caíram de 6 de 8 para 0 de 8
nesse vídeo. As marcações de dano ficaram em 7 de 9. O quadro e a contagem
comparados estão em `evidencias/avaliacao-campo/`.

```powershell
powershell -ExecutionPolicy Bypass -File .\visao\Testar-Camera.ps1 -ObstaculoAteY 0.75
```

O comando restringe o centro das caixas de obstáculo à parte superior do quadro
e desenha a linha de limite. A distância dos objetos e a posição da câmera
determinam a faixa adequada. Sem o argumento, a câmera considera o quadro inteiro.
