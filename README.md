# Calçada Segura · Applied Computer Vision

FIAP · Engenharia de Software 4ES · Global Solution CitizenTech 2026.2

O detector YOLO11n localiza buraco, calçada quebrada e obstáculo em vídeo de calçadas. A aplicação recebe webcam, stream ou MP4, desenha as caixas, grava o resultado e pode enviar ocorrências com foto à API da CitizenTech.

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

## Fechamento e publicação

O relatório final incorpora o treinamento adicional de 19/11/2026, as métricas reais e suas limitações. Os links públicos ficam completos após publicar os arquivos extraídos do pacote. Siga [as instruções de publicação](LEIA-ME-PUBLICACAO.md) antes de enviar o PDF no Teams.

## Novo experimento Roboflow v6

Foram auditadas 389 imagens de uma classe, com correção de quatro grupos de recortes entre partições. O novo treinamento terminou por parada antecipada após 29 épocas. No teste corrigido de 38 imagens, o especialista teve mAP50 de 3,73%, 9 acertos, 64 falsos alarmes e 64 omissões no limiar 0,10 escolhido na validação. Na mesma comparação, o original teve mAP50 de 0,12%. Essas métricas são de uma única classe e de um conjunto diferente do teste histórico de três classes; não comparar diretamente 3,73% com os 36,65% históricos. O especialista permanece experimental e não substitui o modelo padrão.

- [Auditoria, protocolos, pesos e comparação completa](visao/evidencias/roboflow-v6/)
- [Dataset Roboflow corrigido e atribuições](visao/dados/roboflow-v6/)
- [Contexto e prompt para continuidade](CONTEXTO_E_PROMPT_CV.md)

Os dados corrigidos já estão no pacote. Para uma nova rodada preservando a anterior, execute na raiz:

```powershell
python -m pip install -r visao/requirements.txt
python visao/src/retreinar_roboflow.py --etapa treinar --saida visao/evidencias/roboflow-v6-nova-rodada
python visao/src/retreinar_roboflow.py --etapa avaliar --saida visao/evidencias/roboflow-v6-nova-rodada
```

Para repetir apenas o diagnóstico de campo do experimento entregue:

```powershell
python visao/src/avaliar_roboflow_campo.py
```

Para reconstruir o PDF e conferir suas contagens de palavras:

```powershell
python visao/src/gerar_relatorio_final.py
```

Os 44 quadros locais são material de desenvolvimento já inspecionado, com anotações por assistente e sem validação independente de especialista. Não representam teste externo cego.
