"""Reindexa um dataset YOLO já dividido por rota conforme as classes do Calçada Segura."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil

import yaml

from .treinar import CLASSES

EXTENSOES = {".jpeg", ".jpg", ".png", ".webp"}
SPLITS = {"train": "train", "valid": "val", "test": "test"}


def preparar(origem: Path, destino: Path, mapa: dict):
    if not isinstance(mapa, dict) or not mapa or any(v not in CLASSES for v in mapa.values()):
        raise ValueError("O mapa deve associar nomes de origem às classes do Calçada Segura.")
    nomes = yaml.safe_load((origem / "data.yaml").read_text(encoding="utf-8"))["names"]
    if isinstance(nomes, list):
        nomes = dict(enumerate(nomes))
    desconhecidas = mapa.keys() - set(nomes.values())
    if desconhecidas:
        raise ValueError(f"Classes de origem não encontradas: {sorted(desconhecidas)}")
    classes = sorted(set(mapa.values()))
    classes_novas = {i: classes.index(mapa[nome]) for i, nome in nomes.items() if nome in mapa}
    contagens = {}
    hashes = {}
    destino.mkdir(parents=True, exist_ok=False)
    for pasta_origem, split in SPLITS.items():
        base = origem / pasta_origem
        imagens = destino / "images" / split
        labels = destino / "labels" / split
        imagens.mkdir(parents=True)
        labels.mkdir(parents=True)
        contagens[split] = Counter()
        for foto in sorted((base / "images").iterdir()):
            if foto.suffix.lower() not in EXTENSOES:
                continue
            hash_foto = sha256(foto.read_bytes()).hexdigest()
            if hash_foto in hashes:
                raise ValueError(f"Imagem idêntica duplicada: {foto.name} também aparece em {hashes[hash_foto]}.")
            hashes[hash_foto] = f"{pasta_origem}/{foto.name}"
            rotulo = base / "labels" / (foto.stem + ".txt")
            novas_linhas = []
            # Arquivos ausentes são erro. Para negativos revisados, inclua um .txt vazio.
            for linha in rotulo.read_text(encoding="utf-8").splitlines():
                if not linha.strip():
                    continue
                classe, *valores = linha.split()
                classe = int(classe)
                if classe not in nomes or len(valores) != 4:
                    raise ValueError(f"Rótulo de detecção inválido em {rotulo.name}")
                cx, cy, w, h = map(float, valores)
                if not (0 < w <= 1 and 0 < h <= 1
                        and -1e-5 <= cx-w/2 <= cx+w/2 <= 1+1e-5
                        and -1e-5 <= cy-h/2 <= cy+h/2 <= 1+1e-5):
                    raise ValueError(f"Caixa fora da imagem em {rotulo.name}")
                if classe in classes_novas:
                    nova = classes_novas[classe]
                    novas_linhas.append(f"{nova} {' '.join(valores)}")
                    contagens[split][classes[nova]] += 1
            shutil.copy2(foto, imagens / foto.name)
            (labels / rotulo.name).write_text("\n".join(novas_linhas), encoding="utf-8")
            contagens[split]["imagens"] += 1
        faltantes = [classe for classe in classes if not contagens[split][classe]]
        if faltantes:
            raise ValueError(f"Adicione exemplos de {', '.join(faltantes)} ao split {split}.")
    arquivo = {"path": str(destino.resolve()), "train": "images/train", "val": "images/val",
               "test": "images/test", "names": dict(enumerate(classes))}
    (destino / "data.yaml").write_text(yaml.safe_dump(arquivo), encoding="utf-8")
    (destino / "contagens.json").write_text(json.dumps(contagens, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("origem", type=Path)
    parser.add_argument("destino", type=Path)
    parser.add_argument("--mapa", type=Path, required=True)
    args = parser.parse_args()
    preparar(args.origem, args.destino, json.loads(args.mapa.read_text(encoding="utf-8")))
