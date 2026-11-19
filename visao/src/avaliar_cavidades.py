"""Avalia o retreino com rótulos de cavidades revisados sobre dados já reservados."""
import argparse
import json
from pathlib import Path

import cv2
import torch

from .comparar_buracos import BASE, LIMIARES, PESOS, avaliar, carregar


def comparar(dados, checagem, amostra, pesos):
    PESOS["pune_revisado"] = pesos
    padrao = {"perfil": "padrao", "variantes": (BASE,), "limiar": 0.10}
    roma = {"perfil": "roma_480", "variantes": (("retreino", 480),), "limiar": 0.12}
    variantes = [
        {"perfil": "pune_revisado", "variantes": (("pune_revisado", 480),), "limiar": valor}
        for valor in (0.05, 0.06, *LIMIARES)
    ]
    validacao = avaliar(carregar(checagem, "val"), [padrao, roma, *variantes])
    novo = max(validacao[2:], key=lambda r: (r["f1"], -abs(r["limiar"] - roma["limiar"])))
    perfil = {**variantes[0], "limiar": novo["limiar"]}
    fixa = [padrao, roma, perfil]
    dados_saida = {
        "imgsz_pune_revisado": 480,
        "selecao": {"criterio": "Maior F1 na validação revisada. Empates favorecem o limiar mais próximo de 0,12.",
                    "limiar": perfil["limiar"], "validacao": validacao},
        "pune_reservado": avaliar(carregar(checagem, "test"), fixa),
        "teste_publico_original": avaliar(carregar(dados, "test"), fixa),
    }
    if amostra:
        dados_saida["commons_reservado"] = avaliar(carregar(amostra, "test"), fixa)
    return dados_saida


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dados", type=Path, required=True)
    p.add_argument("--checagem", type=Path, required=True)
    p.add_argument("--amostra", type=Path)
    p.add_argument("--pesos", type=Path, required=True)
    p.add_argument("--saida", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    rel = comparar(args.dados, args.checagem, args.amostra, args.pesos)
    args.saida.write_text(json.dumps(rel, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
