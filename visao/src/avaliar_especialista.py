"""Calibra o detector de cavidades em Pune e mede em amostras reservadas."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

import torch
from ultralytics import YOLO

from .calibrar import IOU, caixa_xyxy, contagens

LIMIARES = (.03, .05, .08, .10, .12, .15, .20, .25, .30, .40, .50)
CONJUNTOS = ("val", "test", "sp", "roma")


def cenarios(pasta, modelo, tamanho, device):
    classe = next((i for i, nome in modelo.names.items() if nome == "buraco"), None)
    if classe is None:
        raise ValueError("Os pesos não contêm uma classe buraco.")
    resultados = {}
    for img in sorted(pasta.glob("*.jpg")):
        arquivo_rotulos = pasta.parent.parent / "labels" / pasta.name / img.with_suffix(".txt").name
        anotacoes = [
            (0, caixa_xyxy(tuple(map(float, linha.split()[1:]))))
            for linha in arquivo_rotulos.read_text(encoding="utf-8").splitlines() if linha.strip()
        ]
        dados = modelo.predict(str(img), classes=[classe], conf=LIMIARES[0], imgsz=tamanho,
                               device=device, verbose=False)[0]
        deteccoes = [(0, float(score), caixa) for score, caixa in zip(
            dados.boxes.conf.tolist(), dados.boxes.xyxyn.tolist(), strict=True,
        )]
        resultados[img.name] = anotacoes, deteccoes, dados.speed["inference"]
    return resultados


def medir(casos, limiar, por_ponto=False):
    if not casos:
        raise ValueError("A amostra não contém imagens.")
    total = contagens([(a, p) for a, p, _ in casos.values()], 0, limiar)
    total["imagens"] = len(casos)
    total["ms_inferencia_mediana"] = round(median(t for _, _, t in casos.values()), 2)
    if por_ponto:
        pontos = defaultdict(list)
        for nome, (anotacoes, previsoes, _) in casos.items():
            pontos[nome.split("-", 1)[0]].append((anotacoes, previsoes))
        total["pontos"] = {
            ponto: contagens(itens, 0, limiar) for ponto, itens in sorted(pontos.items())
        }
    return total


def avaliar(pesos, pasta, tamanho, device, limiar_fixo, validacao):
    modelo = YOLO(str(pesos))
    casos_val = cenarios(pasta / "images/val", modelo, tamanho, device)
    candidatos = [{"limiar": t, **medir(casos_val, t)} for t in LIMIARES]
    melhor = max(candidatos, key=lambda x: (x["f1"], x["precisao"], x["limiar"]))
    limiar = limiar_fixo if limiar_fixo is not None else (melhor["limiar"] if melhor["vp"] else .10)
    resultado = {
        "pesos_sha256": sha256(pesos.read_bytes()).hexdigest(), "imgsz": tamanho, "limiar": limiar,
        "limiar_origem": "fixo" if limiar_fixo is not None else "Pune val",
        "val": medir(casos_val, limiar), "curva_val": candidatos,
    }
    if not validacao:
        for split in CONJUNTOS[1:]:
            casos = cenarios(pasta / "images" / split, modelo, tamanho, device)
            resultado[split] = medir(casos, limiar, por_ponto=(split == "sp"))
            resultado[split]["previsoes"] = {
                img: [dict(confianca=round(score, 5), caixa_xyxy=[round(v, 5) for v in caixa])
                      for _, score, caixa in caixas if score >= limiar]
                for img, (_, caixas, _) in casos.items()
            }
    return resultado


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, required=True)
    parser.add_argument("--pesos", type=Path, required=True)
    parser.add_argument("--saida", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limiar-fixo", type=float, choices=LIMIARES)
    parser.add_argument("--somente-validacao", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    resultado = {"iou": IOU, **avaliar(args.pesos, args.dados, args.imgsz, args.device,
                                       args.limiar_fixo, args.somente_validacao)}
    args.saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
