"""Compara ajustes de buraco na validação antes de medir o teste reservado."""
import argparse
import json
from math import sqrt
from pathlib import Path
from time import perf_counter

import cv2
import torch
from ultralytics import YOLO

from .calibrar import caixa_xyxy, contagens, intersecao_sobre_uniao

ARTEFATOS = Path(__file__).resolve().parents[1] / "evidencias"
PESOS = {
    "original": ARTEFATOS / "treino-publico/best.pt",
    "retreino": ARTEFATOS / "reforco-roma/best.pt",
    "especialista": ARTEFATOS / "refino-buraco/best.pt",
}
BASE = ("original", 384)
LIMIARES = (0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
CONF_BRUTA = 0.05
CONSENSOS = (
    (BASE, ("retreino", 384)),
    (BASE, ("especialista", 480)),
    (("retreino", 384), ("especialista", 480)),
    (("retreino", 384), ("retreino", 640)),
    (BASE, ("original", 640)),
)
INDIVIDUAIS = [(nome, tamanho)
               for nome in PESOS for tamanho in (384, 480, 640)]


def combinar(primarias, secundarias):
    confirmadas = []
    pares = []
    for i, (_, c1, b1) in enumerate(primarias):
        for j, (_, c2, b2) in enumerate(secundarias):
            iou = intersecao_sobre_uniao(b1, b2)
            if iou >= 0.35:
                pares.append((iou, sqrt(c1*c2), i, j))
    usados_a, usados_b = set(), set()
    for _, confianca, i, j in sorted(pares, reverse=True):
        if i in usados_a or j in usados_b:
            continue
        usados_a.add(i)
        usados_b.add(j)
        confirmadas.append((0, confianca, primarias[i][2]))
    return confirmadas


def carregar(pasta, split):
    imagens = sorted(f for f in (pasta / "images" / split).iterdir() if f.suffix == ".jpg")
    refs, quadros = [], []
    for f in imagens:
        labels = pasta / "labels" / split / f.with_suffix(".txt").name
        pares = [l.split() for l in labels.read_text().splitlines() if l.strip()]
        refs.append([(int(c), caixa_xyxy(tuple(map(float, b)))) for c, *b in pares if c == "0"])
        img = cv2.imread(str(f))
        if img is None:
            raise ValueError(f"Falha ao decodificar {f}.")
        quadros.append(img)
    return quadros, refs


def obter_previsoes(quadros, variantes):
    resultado, media = {}, {}
    for nome in dict.fromkeys(n for n, _ in variantes):
        modelo = YOLO(str(PESOS[nome]))
        classe = next(i for i, c in modelo.names.items() if c == "buraco")
        for var in (v for v in variantes if v[0] == nome):
            _, tamanho = var
            t0 = perf_counter()
            previsoes = []
            for img in quadros:
                caixas = modelo.predict(img, conf=CONF_BRUTA, imgsz=tamanho,
                                          device="cpu", verbose=False)[0].boxes
                previsoes.append([(0, s, b) for i, s, b in zip(
                    caixas.cls.tolist(), caixas.conf.tolist(), caixas.xyxyn.tolist(),
                    strict=True) if i == classe])
            resultado[var] = previsoes
            media[var] = round(1000 * (perf_counter() - t0) / len(quadros), 2)
            print("predição", var, len(quadros), media[var], "ms/imagem", flush=True)
    return resultado, media


def avaliar(dados, candidatos):
    quadros, refs = dados
    chaves = list(dict.fromkeys(var for cand in candidatos for var in cand["variantes"]))
    previsoes, duracoes = obter_previsoes(quadros, chaves)
    linhas = []
    for cand in candidatos:
        variantes = cand["variantes"]
        if len(variantes) == 1:
            preds = previsoes[variantes[0]]
        else:
            preds = [combinar(p1, p2) for p1, p2 in zip(*(previsoes[v] for v in variantes), strict=True)]
        stats = contagens(list(zip(refs, preds, strict=True)), 0, cand["limiar"])
        chamadas = set((*variantes, BASE))
        linhas.append({**cand, **stats, "passagens": len(chamadas),
                       "ms_modelos_estimados": round(sum(duracoes[v] for v in chamadas), 2)})
    return linhas


def salvar(arquivo, dados):
    arquivo.write_text(json.dumps(dados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validar(dados, saida):
    lista = [{"variantes": vs, "limiar": conf}
             for vs in ([(v,) for v in INDIVIDUAIS] + list(CONSENSOS)) for conf in LIMIARES]
    linhas = avaliar(carregar(dados, "val"), lista)
    padrao = next(v for v in linhas if v["variantes"] == (BASE,) and v["limiar"] == 0.10)
    elegiveis = [v for v in linhas if v["vp"] >= padrao["vp"] and v["fp"] <= padrao["fp"]]
    equilibrado = max(linhas, key=lambda v: (
        v["f1"], -v["passagens"], v["vp"], v["limiar"]))
    conservador = max(elegiveis, key=lambda v: (
        v["f1"], -v["passagens"], v["vp"], v["limiar"]))
    selecao = {"padrao": padrao, "equilibrado": equilibrado, "conservador": conservador}
    salvar(saida / "validacao.json", {"selecao": selecao, "todas": linhas})
    return selecao


def testar(dados, amostra, saida, selecao):
    candidatos = [
        {"perfil": nome, "variantes": tuple(map(tuple, linha["variantes"])), "limiar": linha["limiar"]}
        for nome, linha in selecao.items()
    ]
    teste = avaliar(carregar(dados, "test"), candidatos)
    resultado = {"selecao_validacao": selecao, "teste": teste}
    if amostra:
        resultado["amostra_calcadas"] = avaliar(carregar(amostra, "test"), candidatos)
    salvar(saida / "testes.json", resultado)


def comparar_simulado(dados, amostra, saida, pesos):
    PESOS["simulado"] = pesos
    variantes = (("simulado", 480),)
    original = {"variantes": (BASE,), "limiar": 0.10, "perfil": "padrao"}
    teste_var = [{"variantes": variantes, "limiar": c, "perfil": "simulado"} for c in LIMIARES]
    val = avaliar(carregar(dados, "val"), [original, *teste_var])
    melhor = max((v for v in val if v["perfil"] == "simulado"), key=lambda v: (v["f1"], v["limiar"]))
    avaliados = [original, {**teste_var[0], "limiar": melhor["limiar"]}]
    testemunho = {"validacao_simulado": melhor, "todas_validacao": val,
                 "teste": avaliar(carregar(dados, "test"), avaliados)}
    if amostra:
        testemunho["amostra_calcadas"] = avaliar(carregar(amostra, "test"), avaliados)
    salvar(saida / "testes-simulado.json", testemunho)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dados", type=Path, required=True, help="Conjunto público original")
    p.add_argument("--saida", type=Path, required=True, help="Nova pasta para os resultados")
    p.add_argument("--amostra", type=Path, help="Amostra independente opcional")
    p.add_argument("--pesos-simulado", type=Path, help="Avalia apenas o treino com composições")
    args = p.parse_args()
    args.saida.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    if args.pesos_simulado:
        comparar_simulado(args.dados, args.amostra, args.saida, args.pesos_simulado)
    else:
        selecao = validar(args.dados, args.saida)
        testar(args.dados, args.amostra, args.saida, selecao)
