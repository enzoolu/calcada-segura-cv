"""Confere presença de classes em quadros de campo previamente anotados."""
import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from .inferir import ler_limiares, obstaculo_na_faixa


def avaliar(quadros, anotacoes, pesos, perfil, imgsz, device, obstaculo_ate_y=1.0):
    modelo = YOLO(str(pesos))
    limiares = ler_limiares(perfil, modelo.names, 0.5)
    cenas = anotacoes["cenas"]
    if not cenas:
        raise ValueError("A referência precisa conter ao menos uma cena.")
    resultados = []
    previsoes = {}
    for cena in cenas:
        classe = cena["classe"]
        esperado = cena["presente"]
        if classe not in limiares or type(esperado) is not bool or not cena["quadros"]:
            raise ValueError(f"Referência de cena inválida: {cena['nome']}")
        casos = []
        for caso in cena["quadros"]:
            imagem = caso["imagem"]
            if Path(imagem).name != imagem:
                raise ValueError(f"Informe o nome do quadro, sem diretório: {imagem}")
            if imagem not in previsoes:
                quadro = cv2.imread(str(quadros / imagem))
                if quadro is None:
                    raise ValueError(f"Não foi possível ler o quadro {imagem}.")
                caixas = modelo.predict(quadro, imgsz=imgsz, conf=min(limiares.values()),
                                        device=device, verbose=False)[0].boxes
                maximos = {}
                for c, conf, centro_y in zip(caixas.cls.tolist(), caixas.conf.tolist(),
                                             caixas.xywhn[:, 1].tolist(), strict=True):
                    nome = modelo.names[int(c)]
                    if obstaculo_na_faixa(nome, centro_y, obstaculo_ate_y):
                        maximos[nome] = max(conf, maximos.get(nome, 0))
                previsoes[imagem] = maximos
            maior = previsoes[imagem].get(classe, 0)
            marcou = maior >= limiares[classe]
            casos.append({**caso, "marcou": marcou, "acertou_presenca": marcou == esperado,
                          "confianca_maxima": round(maior, 5) if maior else None})
        resultados.append({
            "nome": cena["nome"], "classe": classe, "presente": esperado,
            "limiar": limiares[classe], "quadros_revisados": len(casos),
            "quadros_com_marcacao": sum(q["marcou"] for q in casos),
            "acertos_de_presenca": sum(q["acertou_presenca"] for q in casos),
            "quadros": casos,
        })
    return {"escopo": "presença por quadro em cenas do mesmo vídeo", "imgsz": imgsz,
            "obstaculo_ate_y": obstaculo_ate_y, "cenas": resultados}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadros", type=Path, required=True)
    parser.add_argument("--referencia", type=Path, required=True)
    parser.add_argument("--pesos", type=Path, required=True)
    parser.add_argument("--perfil", type=Path, required=True)
    parser.add_argument("--saida", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--obstaculo-ate-y", type=float, default=1.0)
    args = parser.parse_args()
    if not 0 < args.obstaculo_ate_y <= 1:
        parser.error("--obstaculo-ate-y precisa estar entre 0 e 1, acima de zero")
    cv2.setNumThreads(2)
    referencias = json.loads(args.referencia.read_text(encoding="utf-8"))
    resultado = avaliar(args.quadros, referencias, args.pesos, args.perfil, args.imgsz,
                        args.device, args.obstaculo_ate_y)
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
