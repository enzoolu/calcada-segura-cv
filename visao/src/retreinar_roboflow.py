"""Audita o ZIP Roboflow v6, treina especialista e compara sem alterar o original.

Exemplo: python visao/src/retreinar_roboflow.py --zip dataset.zip --etapa preparar
Depois: --etapa treinar; --etapa avaliar. O teste é aberto apenas na última etapa.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'visao/dados/roboflow-v6'
OUT = ROOT / 'visao/evidencias/roboflow-v6'
BASE = ROOT / 'visao/evidencias/treino-publico/best.pt'


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare(archive):
    from PIL import Image
    import yaml
    if DATA.exists():
        raise FileExistsError('Dataset já existe; preserve a execução anterior.')
    DATA.mkdir(parents=True)
    OUT.mkdir(parents=True, exist_ok=True)
    records, groups, digests = [], defaultdict(set), set()
    with zipfile.ZipFile(archive) as z:
        config = yaml.safe_load(z.read('data.yaml'))
        if config['names'] != ['Losa_Agrietada']:
            raise ValueError('Classes diferentes do ZIP v6 conferido.')
        for split in ['train', 'valid', 'test']:
            for name in sorted(z.namelist()):
                if not name.startswith(split + '/images/') or not name.endswith('.jpg'):
                    continue
                p = Path(name)
                label_name = split + '/labels/' + p.stem + '.txt'
                image_bytes, label = z.read(name), z.read(label_name).decode('utf-8')
                digest = hashlib.sha256(image_bytes).hexdigest()
                if digest in digests:
                    raise ValueError('Há imagens idênticas; revisar partições antes do treino.')
                digests.add(digest)
                for row in label.splitlines():
                    fields = row.split()
                    if len(fields) != 5 or fields[0] != '0':
                        raise ValueError('Rótulo inesperado: ' + label_name)
                    coords = list(map(float, fields[1:]))
                    if not all(math.isfinite(v) and 0 <= v <= 1 for v in coords) or min(coords[2:]) <= 0:
                        raise ValueError('Coordenada inválida: ' + label_name)
                group = re.split(r'_tile\d+', p.stem)[0]
                groups[group].add(split)
                records.append(dict(name=p.name, original_split=split, group=group,
                                    sha256=digest, boxes=len(label.splitlines()),
                                    source=name, label_source=label_name))
        # Havendo recortes de um quadro em partições diferentes, toda a família
        # vai para a partição mais reservada; nenhum teste é levado ao treino.
        precedence = {'train': 0, 'valid': 1, 'test': 2}
        assigned = {g: max(s, key=precedence.get) for g, s in groups.items()}
        for r in records:
            r['split'] = assigned[r['group']]
            for folder, source in [('images', r['source']), ('labels', r['label_source'])]:
                dest = DATA / r['split'] / folder / Path(source).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(source))
            with Image.open(DATA / r['split'] / 'images' / r['name']) as im:
                im.verify()
        for name in ['README.dataset.txt', 'README.roboflow.txt']:
            (DATA / name).write_bytes(z.read(name))
    (DATA / 'data.yaml').write_text(yaml.safe_dump(dict(path='.', train='train/images',
        val='valid/images', test='test/images', names={0:'calcada_quebrada'}), sort_keys=False))
    audit = dict(zip_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        source='https://universe.roboflow.com/damagedsidewalks/damaged-sidewalks/dataset/6',
        license='CC BY 4.0', original_class='Losa_Agrietada', class_in_project='calcada_quebrada',
        images=len(records), original_groups=len(groups),
        original_cross_split_groups={g:sorted(s) for g,s in groups.items() if len(s)>1},
        corrected_splits={s:dict(images=sum(r['split']==s for r in records),
            boxes=sum(r['boxes'] for r in records if r['split']==s),
            negatives=sum(r['split']==s and r['boxes']==0 for r in records)) for s in precedence},
        limitation='Agrupamento por quadro de origem; não comprova separação por rua, rota ou vídeo.',
        records=records)
    save(OUT / 'auditoria-dataset.json', audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='records'}, ensure_ascii=False, indent=2), flush=True)


def yaml_absolute():
    import yaml
    config=yaml.safe_load((DATA/'data.yaml').read_text())
    config['path']=str(DATA)
    path=OUT/'dados-execucao.yaml'
    path.write_text(yaml.safe_dump(config,sort_keys=False))
    return path


def train(args):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'treino').exists() or (OUT/'best.pt').exists():
        raise FileExistsError('Preserve esta execução: escolha outra pasta com --saida.')
    before=hashlib.sha256(BASE.read_bytes()).hexdigest()
    params=dict(data=str(yaml_absolute()), epochs=args.epochs, imgsz=args.imgsz,
        batch=8, device='cpu', workers=0, seed=42, deterministic=True,
        optimizer='AdamW', lr0=0.001, lrf=0.1, weight_decay=0.0005,
        warmup_epochs=1, freeze=10, patience=10, amp=False, plots=True,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=5, translate=0.05,
        scale=0.2, fliplr=0.5, flipud=0.0, hsv_h=0.01,hsv_s=0.25,hsv_v=0.25,
        project=str(OUT), name='treino', exist_ok=False, cache=False,
        save=True, save_period=-1)
    save(OUT/'protocolo.json',dict(base_sha256=before, params=params,
        selection='best.pt escolhido somente pela validação; teste não usado para ajuste',
        compare='classe 1 do original versus classe 0 do especialista',
        field='44 quadros locais anteriores: diagnóstico de desenvolvimento, não teste externo cego'))
    model=YOLO(str(BASE))
    model.train(**params)
    assert hashlib.sha256(BASE.read_bytes()).hexdigest()==before
    shutil.copyfile(model.trainer.best,OUT/'best.pt')
    print('TREINAMENTO_CONCLUIDO',flush=True)


def ious(a,b):
    import numpy as np
    lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:])
    inter=np.maximum(hi-lo,0).prod()
    return inter/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter,1e-12)


def score(items,threshold):
    tp=fp=fn=0
    for item in items:
        refs=item['reference'];matched=set()
        for p in sorted(item['predictions'],key=lambda p:-p[0]):
            if p[0]<threshold:continue
            options=[(ious(p[1:],r),i) for i,r in enumerate(refs) if i not in matched]
            value,index=max(options,default=(0,None))
            if value>=0.5:matched.add(index);tp+=1
            else:fp+=1
        fn+=len(refs)-len(matched)
    precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
    return dict(tp=tp,fp=fp,fn=fn,precision=precision,recall=recall,
        f1=2*precision*recall/max(precision+recall,1e-12),threshold=threshold)


def collect(model,paths,class_id,imgsz,reference_class=0):
    from PIL import Image
    items=[]
    for p in paths:
        parts=list(p.parts)
        parts[parts.index('images')]='labels'
        label=Path(*parts).with_suffix('.txt')
        refs=[]
        for row in label.read_text().splitlines():
            cls,x,y,w,h=map(float,row.split())
            if int(cls)==reference_class:
                refs.append([x-w/2,y-h/2,x+w/2,y+h/2])
        res=model.predict(str(p),conf=0.001,iou=0.7,imgsz=imgsz,device='cpu',
                          classes=[class_id],verbose=False)[0]
        predictions=[[float(c)]+list(map(float,box)) for c,box in zip(res.boxes.conf,res.boxes.xyxyn.tolist())]
        items.append(dict(image=p.name,reference=refs,predictions=predictions))
    return items


def average_precision(items):
    import numpy as np
    from ultralytics.utils.metrics import ap_per_class
    conf=[];hits=[];targets=[]
    for item in items:
        refs=item['reference'];targets += [0]*len(refs)
        predictions=sorted(item['predictions'],key=lambda p:-p[0])
        matrix=np.zeros((len(predictions),10),dtype=bool)
        for j,t in enumerate(np.arange(0.5,0.96,0.05)):
            taken=set()
            for i,p in enumerate(predictions):
                options=[(ious(p[1:],r),k) for k,r in enumerate(refs) if k not in taken]
                overlap,k=max(options,default=(0,None))
                if overlap>=t:matrix[i,j]=True;taken.add(k)
        conf.extend(p[0] for p in predictions);hits.extend(matrix)
    if not conf or not targets:return dict(map50=0.,map50_95=0.)
    out=ap_per_class(np.asarray(hits),np.asarray(conf),np.zeros(len(conf)),np.asarray(targets),plot=False)
    return dict(map50=float(out[5][:,0].mean()),map50_95=float(out[5].mean()))


def evaluate(args):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(args.threads)
    models={'original':(YOLO(str(BASE)),1),'especialista':(YOLO(str(OUT/'best.pt')),0)}
    report={'scope':'Somente calcada_quebrada. Não comparar este mAP de uma classe com o mAP histórico de três classes.',
            'imgsz':args.imgsz,'iou_match':0.5,'models':{}}
    for name,(model,class_id) in models.items():
        valid=collect(model,sorted((DATA/'valid/images').glob('*.jpg')),class_id,args.imgsz)
        choices=[score(valid,x/100) for x in range(5,91,5)]
        chosen=max(choices,key=lambda s:(s['f1'],s['precision'],s['threshold']))
        # Congela o limiar antes de consultar as predições do teste.
        save(OUT/(name+'-limiar-validacao.json'),dict(chosen=chosen,choices=choices))
        test=collect(model,sorted((DATA/'test/images').glob('*.jpg')),class_id,args.imgsz)
        save(OUT/(name+'-predicoes-validacao.json'),valid)
        save(OUT/(name+'-predicoes-teste.json'),test)
        report['models'][name]=dict(class_id=class_id,validation={**average_precision(valid),**chosen},
            test={**average_precision(test),**score(test,chosen['threshold'])},
            test_common_threshold=score(test,0.30))
    save(OUT/'comparacao.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--etapa',choices=['preparar','treinar','avaliar'],required=True)
    p.add_argument('--zip',type=Path)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--imgsz',type=int,default=416)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--saida',type=Path,default=OUT,help='Pasta de uma execução separada, para preservar a anterior')
    args=p.parse_args()
    OUT=args.saida.resolve()
    if args.etapa=='preparar':
        if args.zip is None:p.error('--zip é obrigatório para preparar')
        prepare(args.zip)
    elif args.etapa=='treinar':train(args)
    else:evaluate(args)
