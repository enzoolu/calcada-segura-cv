"""Gera o PDF da entrega com contagens e resultados conferidos."""
from pathlib import Path
import csv,json,html
from PIL import Image as PILImage,ImageDraw,ImageFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

R=Path(__file__).resolve().parents[2]
DOC=R/'visao/docs';E=R/'visao/evidencias/roboflow-v6'
CONTENT=json.loads((DOC/'conteudo-relatorio.json').read_text())
MET=json.loads((E/'comparacao.json').read_text())
AUD=json.loads((E/'auditoria-dataset.json').read_text())
REPO='https://github.com/enzoolu/calcada-segura-cv'
VIDEO=REPO+'/blob/main/visao/evidencias/demonstracao-visao.mp4'


def example_figure():
    original=json.loads((E/'original-predicoes-teste.json').read_text())
    specialist={x['image']:x for x in json.loads((E/'especialista-predicoes-teste.json').read_text())}
    chosen=[x for x in original if x['reference']][:2]
    sheet=PILImage.new('RGB',(960,720),'white');draw=ImageDraw.Draw(sheet)
    font_path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(font_path),17) if font_path.exists() else ImageFont.load_default()
    for row,item in enumerate(chosen):
        source=R/'visao/dados/roboflow-v6/test/images'/item['image']
        for col,title in enumerate(['Original: confianca 0,30','Especialista: confianca 0,10','Referencia anotada']):
            im=PILImage.open(source).convert('RGB').resize((320,320));d=ImageDraw.Draw(im)
            if col==2:boxes=item['reference'];color=(10,170,70)
            else:
                pred=item if col==0 else specialist[item['image']]
                boxes=[x[1:] for x in pred['predictions'] if x[0]>=(0.30 if col==0 else 0.10)]
                color=(0,90,255)
            for x1,y1,x2,y2 in boxes:d.rectangle((x1*320,y1*320,x2*320,y2*320),outline=color,width=3)
            sheet.paste(im,(col*320,row*360+30));draw.text((col*320+5,row*360+4),title,font=font,fill='black')
    path=E/'comparacao-visual.jpg';sheet.save(path,quality=92);return path


def main():
    assert len(CONTENT['resumo'].split())==250
    assert len(CONTENT['objetivo'].split())==50
    normal='Helvetica';bold='Helvetica-Bold'
    for a,b in [('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),('C:/Windows/Fonts/arial.ttf','C:/Windows/Fonts/arialbd.ttf')]:
        if Path(a).exists() and Path(b).exists():
            pdfmetrics.registerFont(TTFont('Body',a));pdfmetrics.registerFont(TTFont('BodyBold',b));normal='Body';bold='BodyBold';break
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyTextCV',fontName=normal,fontSize=9.4,leading=14,spaceAfter=8,textColor=colors.HexColor('#253347')))
    styles.add(ParagraphStyle(name='TitleCV',fontName=bold,fontSize=28,leading=34,textColor=colors.HexColor('#123C47'),spaceAfter=18))
    styles.add(ParagraphStyle(name='HeadingCV',fontName=bold,fontSize=16,leading=21,spaceBefore=9,spaceAfter=12,textColor=colors.HexColor('#126B6D')))
    styles.add(ParagraphStyle(name='SmallCV',fontName=normal,fontSize=8,leading=11,spaceAfter=8))
    story=[]
    def p(text,small=False):story.append(Paragraph(text,styles['SmallCV' if small else 'BodyTextCV']))
    def h(text):story.append(Paragraph(text,styles['HeadingCV']))
    def table(rows,widths=None):
        cells=[[Paragraph(str(c),styles['SmallCV']) for c in row] for row in rows]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E3F1EF')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),0.8,colors.HexColor('#126B6D')),('GRID',(0,1),(-1,-1),0.35,colors.HexColor('#D3DCDF')),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.extend([t,Spacer(1,10)])
    def picture(path,width=490,maxheight=300):
        w,ht=PILImage.open(path).size;scale=min(width/w,maxheight/ht);story.append(Image(str(path),width=w*scale,height=ht*scale));story.append(Spacer(1,6))
    def link(label,url):p(f'<b>{html.escape(label)}</b><br/><link href="{html.escape(url,quote=True)}" color="#126B6D">{html.escape(url)}</link>',True)
    def page():story.append(PageBreak())
    p('FIAP | GLOBAL SOLUTION 2026.2 | CITIZENTECH',True);story.append(Spacer(1,30))
    story.append(Paragraph('Calçada Segura',styles['TitleCV']))
    h('Detecção de defeitos em passeios urbanos')
    p('Applied Computer Vision - Engenharia de Software, 4º ano<br/>Entrega técnica - 19 de novembro de 2026')
    story.append(Spacer(1,18))
    table([['Integrante','RM'],['Enzo Luiz Goulart','99666'],['João Pedro Cruz','98650'],['Tiago Paulino','551169'],['Victor Eid','98668']],[360,130])
    p('<b>Escopo:</b> detector customizado de três classes, demonstração em vídeo e estudo adicional de treinamento de um especialista em calçadas trincadas.')
    p('<b>Decisão técnica:</b> manter o detector original como padrão do protótipo. O novo especialista permanece experimental; suas limitações e resultados estão documentados.')
    link('Repositório e documentação',REPO)
    link('Vídeo principal da solução - 1 min 28 s',VIDEO)
    page()
    h('Resumo - 250 palavras');p(html.escape(CONTENT['resumo']))
    h('Introdução')
    p('O Calçada Segura propõe apoiar o registro e a triagem de problemas no passeio público. O componente visual identifica regiões candidatas em imagens e pode encaminhar ocorrências com foto para uma API. A saída deve apoiar revisão humana: uma detecção não certifica acessibilidade, mede profundidade ou determina sozinha a gravidade de um dano.')
    p('A heterogeneidade das calçadas é central ao problema. Materiais, juntas, padrões de piso tátil, remendos, sombras e objetos podem alterar a aparência de um defeito ou imitar sua forma. O protótipo exige avaliação explícita de falsos positivos e omissões, além de exemplos visualmente convincentes.')
    h('Objetivo - 50 palavras');p(html.escape(CONTENT['objetivo']))
    page()
    h('Ferramentas e funcionamento')
    table([['Ferramenta','Papel'],['Python 3.12 e Ultralytics 8.4.212','Treinamento customizado, inferência e avaliação YOLO11n.'],['PyTorch 2.15.0+cpu / torchvision 0.29.2+cpu','Execução do novo treinamento em CPU.'],['OpenCV e Pillow','Entrada e gravação de vídeo, processamento e exemplos visuais.'],['HTTPX, JSON e UUID','Envio de ocorrências e fila local com evidência fotográfica.'],['ReportLab e unittest','Geração reproduzível do PDF e verificação de confirmação/envio.']],[160,330])
    p('Fluxo implementado: câmera/MP4 -> detector -> limiares por classe -> confirmação temporal -> fila com foto -> API. A integração documentada anteriormente continua local; não é apresentada como implantação comprovada no MVP remoto da equipe.')
    h('Metodologia - preparação dos dados')
    p('1. Preservar os pesos originais e inventariar código, vídeo, métricas e testes. 2. Abrir o ZIP Roboflow v6 e conferir imagens, rótulos, classes, licença e hashes. O pacote contém 389 imagens de 640 x 640 px, uma classe e 613 caixas; não foram encontrados rótulos inválidos ou imagens idênticas por SHA256. Rótulos vazios são negativos válidos.')
    p('3. Reunir os recortes de cada imagem de origem na mesma partição. Quatro grupos estavam divididos: 3228, 3242, 3468 e 2435. A correção preservou a precedência teste &gt; validação &gt; treino, sem levar imagens de teste para treino. Essa separação não comprova independência por rua, rota ou gravação.')
    table([['Partição corrigida','Imagens','Caixas','Negativos'],['Treino','275','422','167'],['Validação','76','118','45'],['Teste','38','73','22']],[250,80,80,80])
    p('O nome de exportação YOLOv8 representa o formato das anotações. A arquitetura utilizada continua YOLO11n. A classe Losa_Agrietada foi mapeada para calcada_quebrada; o ZIP não contém classes de buraco ou obstáculo.',True)
    page()
    h('Metodologia - treinamento e avaliação')
    p('4. Ajustar um especialista de uma classe a partir do checkpoint do projeto, sem substituir o original de três classes. Configuração: até 30 épocas, 416 px, batch 8, AdamW, taxa inicial 0,001, primeiras 10 camadas congeladas, seed 42, mosaic desativado e aumento moderado. A parada antecipada encerrou o registro em 29 épocas. O melhor checkpoint foi selecionado somente pela validação.')
    p('5. Comparar ambos nos mesmos conjuntos e resolução. Mapear classe 1 do original e classe 0 do especialista para calçada quebrada. Selecionar o limiar pela validação e congelá-lo antes do teste. Calcular correspondências individuais com IoU de pelo menos 0,50, AP e métricas operacionais. As predições, os limiares e os protocolos foram preservados em JSON.')
    p('6. Conferir também 44 quadros locais já utilizados no desenvolvimento e executar os oito testes de confirmação, filtragem, fila e tratamento de falhas. Esses quadros não constituem teste externo cego; várias imagens são vistas dos mesmos pontos.')
    h('Resultados do detector principal - evidências anteriores')
    table([['Avaliação histórica','Resultado'],['Teste público, 112 imagens e três classes','mAP50: 36,65%; mAP50-95: 21,77%.'],['Campo, piso tátil danificado','Marcação em 7 de 9 quadros selecionados.'],['Cavidades paulistanas, 16 quadros','0 de 7 caixas localizadas na avaliação anterior.']],[275,215])
    p('Os resultados históricos acima foram recuperados do pacote anterior. Não são uma nova medição nesta execução nem devem ser comparados diretamente com o teste Roboflow de uma única classe e imagens diferentes.',True)
    h('Novo experimento - mesmo teste Roboflow corrigido')
    a=MET['models']['original']['test'];b=MET['models']['especialista']['test']
    pct=lambda x:f'{100*x:.2f}%'.replace('.',',')
    table([['Métrica (38 imagens, uma classe)','Original','Especialista'],['mAP50',pct(a['map50']),pct(b['map50'])],['mAP50-95',pct(a['map50_95']),pct(b['map50_95'])],['Confiança selecionada na validação',str(a['threshold']),str(b['threshold'])],['Acertos / falsos alarmes / omissões',f"{a['tp']} / {a['fp']} / {a['fn']}",f"{b['tp']} / {b['fp']} / {b['fn']}"],['Precisão / recall / F1',f"{pct(a['precision'])} / {pct(a['recall'])} / {pct(a['f1'])}",f"{pct(b['precision'])} / {pct(b['recall'])} / {pct(b['f1'])}"]],[270,110,110])
    p('O especialista acertou 9 das 73 caixas, mas produziu 64 falsos alarmes no limiar 0,10. No limiar comum 0,30, ambos tiveram zero acertos: o original teve 11 falsos alarmes e o especialista zero. A melhora de AP neste conjunto permanece pequena e não justifica declarar prontidão para uso.',True)
    p('No diagnóstico dos mesmos 44 quadros locais, o mAP50 de calçada quebrada foi 34,18% no original e 0,11% no especialista. No limiar comum 0,30, o original teve 16 acertos, 24 falsos alarmes e 17 omissões; o especialista, zero acertos, 4 falsos alarmes e 33 omissões. São quadros de desenvolvimento já examinados, não um teste externo cego. Essa perda adicional sustenta a decisão de preservar o padrão original.',True)
    page()
    h('Exemplos de resultados e demonstração')
    picture(example_figure(),maxheight=365)
    p('Figura 1. Duas imagens de teste: original à esquerda, especialista ao centro e referência à direita. Caixas azuis são predições; verdes são anotações de referência, mostradas exclusivamente na terceira coluna. Os limiares diferentes estão identificados nos painéis. A figura permite observar omissões e falsos alarmes, sem ocultar erros.',True)
    picture(R/'integracao/evidencias/campo/quadro-detectado.jpg',maxheight=165)
    p('Figura 2. Evidência de campo preservada do projeto anterior. O vídeo principal demonstracao-visao.mp4, de 87,95 segundos, apresenta o detector original. É uma demonstração offline sobre gravação; a taxa de reprodução do arquivo não comprova inferência em tempo real. O programa também aceita webcam ou stream no computador de execução.',True)
    page()
    h('Limitações, generalização e próximos passos')
    p('Os novos dados cobrem apenas uma classe e não fornecem supervisão de buracos ou obstáculos. Existem muitos negativos e regiões de dano pequenas. Imagens da mesma captura ou de pontos próximos podem compartilhar aparência mesmo quando seus hashes diferem. O agrupamento realizado reduz uma forma identificável de sobreposição, mas não elimina toda correlação entre cenas.')
    p('Diferenças entre materiais, acabamento, conservação, luz, sombra, perspectiva, distância e escala tornam plausível uma mudança de domínio entre o dataset público e calçadas paulistanas. Essa interpretação é compatível com as falhas observadas; não foi demonstrada como causa isolada por um experimento controlado.')
    p('Melhorar exige cobertura representativa desses cenários e revisão consistente das anotações, além de exemplos negativos difíceis. Uma coleção maior pode ajudar, mas volume isolado não garante qualidade. Recomenda-se coletar locais e defeitos físicos variados, reservar ruas ou rotas independentes antes do treino e avaliar por classe, escala, iluminação e tipo de piso. Mais quadros do mesmo buraco não equivalem a mais buracos independentes.')
    h('Reprodução e testes')
    p('O repositório inclui código, pesos original e experimental, dataset Roboflow corrigido, atribuição, manifesto, configurações, curva de treino, predições e resultados. Os oito testes automatizados existentes passaram nesta execução. O PDF é gerado por visao/src/gerar_relatorio_final.py, que verifica resumo de 250 e objetivo de 50 palavras.')
    p('O arquivo LEIA-ME-PUBLICACAO.md descreve extração na pasta clonada, commit, push e verificação pública. O README apresenta os comandos para câmera, vídeo e reprodução do experimento. A publicação é feita pelo responsável pela entrega antes do envio do PDF ao Teams.')
    h('Conclusão')
    p('O protótipo implementa detecção customizada e registro de evidências, mas apresenta omissões relevantes. O treinamento adicional foi executado, auditado e comparado; sua melhora limitada no novo conjunto não eliminou falsos alarmes nem comprova generalização urbana. Mantém-se o modelo original como padrão do protótipo e o especialista como resultado experimental. A entrega torna verificáveis tanto o funcionamento quanto suas limitações.')
    h('Referências e atribuições')
    link('Dataset DamagedSidewalks v6 - CC BY 4.0','https://universe.roboflow.com/damagedsidewalks/damaged-sidewalks/dataset/6')
    link('Ultralytics - treinamento e formato de detecção','https://docs.ultralytics.com/modes/train/')
    p('Demais fontes públicas, licenças e manifestos históricos estão em visao/evidencias/treino-publico/fontes.md e nas pastas de evidências. Os quadros locais e vídeos foram fornecidos pelo integrante responsável pelo projeto.',True)
    out=DOC/'relatorio-visao.pdf'
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#D3DCDF'));canvas.line(48,42,547,42);canvas.setFont(normal,7.5);canvas.setFillColor(colors.HexColor('#526272'));canvas.drawString(48,29,'Calçada Segura | Applied Computer Vision | GS 2026.2');canvas.drawRightString(547,29,str(doc.page))
    SimpleDocTemplate(str(out),pagesize=(595.28,841.89),leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=54,title='Calçada Segura - Applied Computer Vision',author='Enzo Luiz Goulart; João Pedro Cruz; Tiago Paulino; Victor Eid').build(story,onFirstPage=footer,onLaterPages=footer)
    print(out)


if __name__=='__main__':main()
