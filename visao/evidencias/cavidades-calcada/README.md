# Avaliação de cavidades no passeio

Uma cavidade tem abertura visível ou perda localizada de material com rebaixo no passeio. As imagens sem essa evidência foram selecionadas entre trechos livres, juntas, grelhas fechadas, reparos, pisos apenas rachados e outras texturas. A revisão procurou reconhecer e separar vistas repetidas de uma mesma cena. Dois pares de Pune tiveram distância de dHash igual a 6 e foram conferidos visualmente como cenas distintas.

| Conjunto | Fotos ou quadros | Caixas ou uso |
| --- | --- | --- |
| Pune treino inicial | 53 | 21 caixas manuais |
| Pune treino ampliado | 71 | 25 caixas manuais e 14 recortes gerados a partir do treino |
| Pune validação | 20 | 6 caixas manuais |
| Pune teste | 17 | 6 caixas manuais |
| São Paulo teste | 16 | 7 caixas manuais em 3 pontos com cavidade, 4 gravações no total |
| Roma treino e teste | 42 em cada parte | Rótulos públicos de vias, sendo 44 caixas no teste |

`manifesto-pune-roma.json` e `manifesto-ampliado.json` identificam as imagens, a origem, a divisão e as caixas. `referencia-sp.json` identifica as gravações e a ordem dos quadros extraídos com FFmpeg. Cada ponto usa a versão transcodificada identificada pelo SHA-256. Nenhum vídeo paulistano integra o treino.

Os dois retreinos foram calibrados na validação de Pune. A melhor variante pelo F1 nessa validação localizou 1 de 6 cavidades com confiança 0,50. O limiar foi mantido para o teste. Nos quadros de São Paulo, a variante escolhida e o detector geral localizaram 0 de 7 cavidades. A tabela na seção **Cavidades no passeio** do PDF compara também os falsos alarmes. `comparacao.json` guarda as caixas previstas e as contagens por ponto. Os quadros de um mesmo ponto compartilham a cena.

## Reproduzir

Use o ZIP oficial de Pune (cerca de 4,5 GB) e o arquivo `archive.zip` de Roma. Se os JPEGs de Pune já estiverem pré-processados em pastas Train, Val e Test, use `--imagens-pune` no lugar de `--pune-zip`. As imagens de Pune passam por correção de orientação, redimensionamento para no máximo 1280 px e codificação JPEG. Variações no codificador podem alterar o hash do JPEG. As imagens de Roma são verificadas byte a byte.

Para repetir o teste de São Paulo, coloque as versões transcodificadas indicadas no manifesto em `fontes/sp` com os nomes `sp01.mov`, `sp02.mov`, `sp03.mov` e `sp05.mov`. FFmpeg deve estar disponível no PATH.

```powershell
py -3.12 -m visao.src.preparar_especialista --manifesto visao/evidencias/cavidades-calcada/manifesto-ampliado.json --pune-zip fontes/pune.zip --roma-zip fontes/archive.zip --referencia-sp visao/evidencias/cavidades-calcada/referencia-sp.json --videos-sp fontes/sp --recortar-treino --saida dados/cavidades
py -3.12 -m visao.src.avaliar_especialista --dados dados/cavidades --pesos visao/evidencias/cavidades-calcada/pesos-ampliados.pt --imgsz 480 --limiar-fixo 0.50 --saida comparacao-reproducao.json
py -3.12 -m visao.src.treinar_especialista --dados dados/cavidades/data.yaml --saida runs/cavidades-ampliadas --epocas 28 --imgsz 512 --congelar 3
```

Para o primeiro retreino, prepare outro diretório com `manifesto-pune-roma.json` e sem `--recortar-treino`. Treine com os valores padrão do script e teste `pesos-pune.pt` a 480 px e limiar 0,15. As curvas de treino estão nos CSVs desta pasta.

## Créditos

- Pune. Chakurkar et al., Indian Footpath Damage Segmentation Dataset, CC BY 4.0. Caixas manuais adicionadas à seleção e imagens redimensionadas. [Zenodo](https://doi.org/10.5281/zenodo.17154852)
- Roma. Giordani et al., Road Damage Dataset, CC BY 4.0. Seleção e conversão dos rótulos para a classe única. [Zenodo](https://doi.org/10.5281/zenodo.17834373)
- São Paulo. Títulos, ligações às páginas de autoria e marcações de cada gravação estão em [referencia-sp.json](referencia-sp.json). O pacote contém a referência textual e as previsões por quadro.
