"""Gera composições para treino com recortes anotados e pisos reais já disponíveis."""
import argparse
from hashlib import sha256
import json
from math import cos, pi, sin
from pathlib import Path
from random import Random
from shutil import copytree
from zipfile import ZipFile

import cv2
import numpy as np
import yaml

from .calibrar import caixa_xyxy


def recortes(arquivo, manifesto):
    dados = json.loads(manifesto.read_text(encoding="utf-8"))
    with ZipFile(arquivo) as base:
        for item in dados["dados_exclusivos_de_treino"]:
            if not item["caixas"]:
                continue
            nome = item["origem"]
            original = base.read(nome)
            if sha256(original).hexdigest() != item["sha256"]:
                raise ValueError(f"Imagem diverge do manifesto de treino: {nome}")
            imagem = cv2.imdecode(np.frombuffer(original, dtype=np.uint8), cv2.IMREAD_COLOR)
            lab = nome.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            altura, largura = imagem.shape[:2]
            for i, linha in enumerate(base.read(lab).decode("utf-8-sig").splitlines()):
                classe, *valores = linha.split() if linha.strip() else (None,)
                if classe != "0" or len(valores) < 6 or len(valores) % 2:
                    continue
                pontos = np.asarray(valores, dtype=float).reshape(-1, 2) * (largura, altura)
                pontos = np.rint(np.clip(pontos, (0, 0), (largura-1, altura-1))).astype(np.int32)
                x, y, w, h = cv2.boundingRect(pontos)
                if min(w, h) < 12 or cv2.contourArea(pontos) < 90:
                    continue
                mascara = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mascara, [pontos - (x, y)], 255)
                yield nome, i, imagem[y:y+h, x:x+w].copy(), mascara


def compor(piso, caixas, recorte, mascara, rng):
    altura, largura = piso.shape[:2]
    escala = rng.uniform(0.10, 0.32) * min(altura, largura) / max(mascara.shape)
    w = max(10, round(mascara.shape[1] * escala))
    h = max(10, round(mascara.shape[0] * escala * rng.uniform(0.8, 1.2)))
    min_x = max(2, round(largura*0.14 - w/2))
    max_x = min(largura-w-2, round(largura*0.86 - w/2))
    min_y = max(2, round(altura*0.43 - h/2))
    max_y = min(altura-h-2, round(altura*0.87 - h/2))
    if min_x > max_x or min_y > max_y:
        return None
    for _ in range(30):
        x, y = rng.randint(min_x, max_x), rng.randint(min_y, max_y)
        bbox = (x/largura, y/altura, (x+w)/largura, (y+h)/altura)
        distancias = [max(a[0]-bbox[2], bbox[0]-a[2], a[1]-bbox[3], bbox[1]-a[3])
                      for a in caixas]
        if distancias and 0 <= min(distancias) <= 0.23:
            break
    else:
        return None
    recorte = cv2.resize(recorte, (w, h), interpolation=cv2.INTER_LINEAR)
    mascara = cv2.resize(mascara, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        recorte, mascara = cv2.flip(recorte, 1), cv2.flip(mascara, 1)
    contorno = np.zeros((h, w), dtype=np.uint8)
    pontos = []
    for i in range(12):
        angulo = i*pi/6
        raio = rng.uniform(0.68, 1)
        pontos.append((round((1 + raio*cos(angulo))*(w-1)/2),
                       round((1 + raio*sin(angulo))*(h-1)/2)))
    cv2.fillPoly(contorno, [np.asarray(pontos, dtype=np.int32)], 255)
    mascara = cv2.bitwise_and(mascara, contorno)
    interior = recorte[mascara >= 127]
    if not interior.size:
        return None
    duro = (mascara >= 127).astype(np.uint8)
    distancia = cv2.distanceTransform(np.pad(duro, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    alfa = np.clip(distancia / max(2, 0.14*min(w, h)), 0, 1)[:, :, None]
    fundo = piso[y:y+h, x:x+w]
    # Ajusta a cor do recorte parcialmente, preservando a textura e sombras da cavidade.
    ajuste = np.clip(fundo.mean(axis=(0, 1)) - interior.mean(axis=0), -36, 36) * 0.30
    fundo[:] = np.clip(recorte * alfa + fundo * (1-alfa) + ajuste * alfa, 0, 255).astype(np.uint8)
    dx, dy, lw, lh = cv2.boundingRect(duro)
    return ((x+dx+lw/2)/largura, (y+dy+lh/2)/altura, lw/largura, lh/altura)


def preparar(base, arquivo, manifesto, saida, limite):
    if limite < 1:
        raise ValueError("A quantidade deve ser positiva.")
    rng = Random(42)
    partes = list(recortes(arquivo, manifesto))
    if not partes:
        raise ValueError("O manifesto não corresponde a recortes de treino disponíveis.")
    copytree(base, saida, ignore=lambda _, nomes: [n for n in nomes if n.endswith(".cache")])
    configuracao = yaml.safe_load((base / "data.yaml").read_text(encoding="utf-8"))
    configuracao["path"] = str(saida.resolve())
    (saida / "data.yaml").write_text(yaml.safe_dump(configuracao), encoding="utf-8")
    pisos = sorted((base / "images/train").glob("pune-*.jpg"))
    rng.shuffle(pisos)
    itens = []
    for arq in pisos:
        piso = cv2.imread(str(arq))
        if piso is None:
            raise ValueError(f"Não foi possível abrir {arq}")
        lab = base / "labels/train" / arq.with_suffix(".txt").name
        linhas = lab.read_text(encoding="utf-8").strip().splitlines()
        caixas = [caixa_xyxy(tuple(map(float, linha.split()[1:]))) for linha in linhas]
        for origem, poligono, recorte, mascara in rng.sample(partes, min(4, len(partes))):
            caixa = compor(piso, caixas, recorte, mascara, rng)
            if caixa is not None:
                break
        else:
            continue
        nome = f"cavidade-simulada-{arq.stem}"
        destino = saida / "images/train" / f"{nome}.jpg"
        if not cv2.imwrite(str(destino), piso, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise OSError(f"Não foi possível gravar {destino}")
        linhas.append("0 " + " ".join(f"{valor:.7f}" for valor in caixa))
        (saida / "labels/train" / f"{nome}.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")
        itens.append({"imagem": destino.name, "piso_de_treino": arq.name, "origem_do_recorte": origem,
                      "poligono": poligono, "sha256": sha256(destino.read_bytes()).hexdigest()})
        if len(itens) == limite:
            break
    if not itens:
        raise ValueError("Nenhum piso de treino tem espaço livre para as cavidades.")
    (saida / "simuladas.json").write_text(json.dumps(itens, indent=2) + "\n", encoding="utf-8")
    print(f"{len(itens)} imagens compostas com caixas de buraco, somente no treino.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--roma", type=Path, required=True)
    parser.add_argument("--manifesto", type=Path, required=True)
    parser.add_argument("--saida", type=Path, required=True)
    parser.add_argument("--limite", type=int, default=100)
    parser.add_argument("--pesos", type=Path, help="Pesos de partida para o treino opcional")
    parser.add_argument("--treino", type=Path, help="Pasta nova para os pesos ajustados")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if (args.pesos is None) != (args.treino is None):
        parser.error("informe --pesos e --treino juntos")
    preparar(args.base, args.roma, args.manifesto, args.saida, args.limite)
    if args.treino:
        from .refinar_buracos import ajustar

        ajustar(args.pesos, args.saida / "data.yaml", args.treino, args.device,
                epocas=6, imgsz=448, batch=12, lr=0.00045)
