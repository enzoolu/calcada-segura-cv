# Calçada Segura · Applied Computer Vision

FIAP · Engenharia de Software 4ES · Global Solution CitizenTech 2026.2

O detector YOLO11n localiza buraco, calçada quebrada e obstáculo em vídeo de calçadas. A aplicação recebe webcam, stream ou MP4, desenha as caixas, grava o resultado e pode enviar ocorrências com foto à API própria do protótipo.

| Integrante | RM |
| --- | --- |
| Enzo Luiz Goulart | 99666 |
| João Pedro Cruz | 98650 |
| Tiago Paulino | 551169 |
| Victor Eid | 98668 |

## Entrega

- [Relatório técnico](visao/docs/relatorio-visao.pdf)
- [Vídeo da solução, 1 min 28 s](visao/evidencias/demonstracao-visao.mp4)
- [Código, treinamento e instruções](visao/README.md)
- [Resultados e limiares do modelo principal](visao/evidencias/treino-publico/)
- [Teste de cavidades com dados reservados](visao/evidencias/cavidades-calcada/)

## Executar no Windows

Com Python 3.12 instalado, abra o PowerShell na raiz do repositório e execute.

```powershell
powershell -ExecutionPolicy Bypass -File .\visao\Testar-Camera.ps1
```

Para usar o trecho gravado incluído no repositório.

```powershell
powershell -ExecutionPolicy Bypass -File .\visao\Testar-Camera.ps1 -Fonte .\integracao\evidencias\campo\trecho.mp4
```

A tecla Q encerra a janela. A saída anotada fica em `saida/`. A opção `-ObstaculoAteY 0.75` restringe o centro das caixas de obstáculo à faixa superior de 75% da imagem. Ela foi conferida em cenas do vídeo de campo e pode omitir obstáculos reais na parte inferior.

## Resultado medido

O modelo principal obteve mAP50 de 36,65% nas 112 imagens públicas de teste. No vídeo de campo, marcou o piso tátil danificado em 7 dos 9 quadros selecionados. No teste de cavidades paulistanas, localizou 0 de 7 caixas anotadas em 16 quadros de quatro gravações históricas. O PDF documenta as omissões e os retreinos. Esses quadros compartilham cenas de um mesmo ponto.

## Testes

Na pasta `visao`, com as dependências instaladas.

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -m unittest discover -s tests -v
```

O vídeo demonstra inferência sobre gravação de celular. As coordenadas usadas nas evidências de integração são um ponto de teste na Sé. Os créditos e as licenças das bases estão no relatório e nos manifestos.
## Repositórios da equipe

- [Visão Computacional](https://github.com/enzoolu/calcada-segura-cv)
- [RPA](https://github.com/enzoolu/calcada-segura-rpa)
