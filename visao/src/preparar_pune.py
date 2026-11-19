"""Converte máscaras binárias do dataset de Pune em caixas de calçada quebrada."""
import argparse
import json
from pathlib import Path

import cv2
import yaml

SPLITS = {"Train": "train", "Val": "val", "Test": "test"}


def preparar(origem, destino):
    contagens = {}
    for original, split in SPLITS.items():
        imagens, labels = (destino / pasta / split for pasta in ("images", "labels"))
        imagens.mkdir(parents=True, exist_ok=False)
        labels.mkdir(parents=True, exist_ok=False)
        conjunto = sorted((origem / original).glob("*.jpg"))
        if not conjunto:
            raise ValueError(f"Sem imagens em {origem / original}")
        caixas = 0
        for foto in conjunto:
            imagem = cv2.imread(str(foto))
            mascara = cv2.imread(str(foto.with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
            if imagem is None or mascara is None:
                raise ValueError(f"Imagem ou máscara ilegível: {foto.name}")
            largura, altura = imagem.shape[1], imagem.shape[0]
            escala = min(1, 960 / max(largura, altura))
            dimensoes = (round(largura * escala), round(altura * escala))
            imagem = cv2.resize(imagem, dimensoes, interpolation=cv2.INTER_AREA)
            mascara = cv2.resize(mascara, dimensoes, interpolation=cv2.INTER_NEAREST)
            n, _, stats, _ = cv2.connectedComponentsWithStats((mascara > 0).astype("uint8"), connectivity=8)
            linhas = []
            for x, y, w, h, area in stats[1:n]:
                if area < 42 or w < 5 or h < 5:
                    continue
                larg, alt = dimensoes
                cx, cy = (x + w / 2) / larg, (y + h / 2) / alt
                linhas.append(f"0 {cx:.6f} {cy:.6f} {w/larg:.6f} {h/alt:.6f}")
            if not cv2.imwrite(str(imagens / foto.name), imagem):
                raise OSError(f"Falha ao gravar {foto.name}")
            (labels / f"{foto.stem}.txt").write_text("\n".join(linhas), encoding="utf-8")
            caixas += len(linhas)
        contagens[split] = {"imagens": len(conjunto), "caixas": caixas}
    (destino / "data.yaml").write_text(yaml.safe_dump({
        "path": str(destino.resolve()), "train": "images/train", "val": "images/val",
        "test": "images/test", "names": {0: "calcada_quebrada"},
    }), encoding="utf-8")
    (destino / "contagens.json").write_text(json.dumps(contagens, indent=2), encoding="utf-8")
    print(json.dumps(contagens, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("origem", type=Path, help="pasta contendo Train, Val e Test")
    parser.add_argument("destino", type=Path, help="pasta nova para o formato YOLO")
    args = parser.parse_args()
    preparar(args.origem, args.destino)
