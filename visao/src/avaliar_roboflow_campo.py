"""Diagnóstico do especialista nos 44 quadros locais já examinados anteriormente."""
from pathlib import Path
import json
import torch
from ultralytics import YOLO
from retreinar_roboflow import ROOT,OUT,BASE,collect,score,average_precision,save

if __name__=='__main__':
    torch.set_num_threads(4)
    paths=sorted((ROOT/'visao/dados/campo-local/images').rglob('*.jpg'))
    if len(paths)!=44:
        raise ValueError('São esperados 44 quadros; preserve o conjunto completo.')
    report=json.loads((OUT/'comparacao.json').read_text())
    for name,peso,cls in [('original',BASE,1),('especialista',OUT/'best.pt',0)]:
        items=collect(YOLO(str(peso)),paths,cls,416,reference_class=1)
        threshold=report['models'][name]['validation']['threshold']
        report['models'][name]['field_development']={
            **average_precision(items),**score(items,threshold),
            'common_threshold':score(items,0.30),'images':len(paths),
            'scope':'Quadros locais já examinados. Diagnóstico de desenvolvimento; não é teste externo cego.'}
        save(OUT/(name+'-predicoes-campo.json'),items)
    save(OUT/'comparacao.json',report)
    print(json.dumps({n:v['field_development'] for n,v in report['models'].items()},indent=2))
