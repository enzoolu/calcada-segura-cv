"""Monta um conjunto YOLO com três classes e splits auditáveis via manifesto."""
import argparse
import csv
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

import yaml

CLASSES = ("buraco", "calcada_quebrada", "obstaculo")
SPLITS = ("train", "val", "test")


def preparar(arquivo: Path, bases: Path, destino: Path):
    with arquivo.open(encoding="utf-8-sig", newline="") as stream:
        itens = list(csv.DictReader(stream))
    if not itens:
        raise ValueError("Manifesto vazio.")
    destino.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        for pasta in ("images", "labels"):
            (destino / pasta / split).mkdir(parents=True)
    hashes = {}
    for item in itens:
        split, origem = item["split"], item["fonte"]
        caminho = bases / item["imagem"]
        nome = f"{origem}-{caminho.name}"
        assinatura = sha256(caminho.read_bytes()).hexdigest()
        anterior = hashes.get(assinatura)
        if anterior:
            raise ValueError(f"Imagem duplicada em {anterior} e {item['imagem']}.")
        if assinatura != item["sha256"]:
            raise ValueError(f"O conteúdo de {item['imagem']} diverge do manifesto.")
        hashes[assinatura] = item["imagem"]
        alvo = destino / "images" / split / nome
        if alvo.exists():
            raise ValueError(f"Nome de imagem repetido em {split}: {nome}.")
        copyfile(caminho, alvo)
        (destino / "labels" / split / Path(nome).with_suffix(".txt")).write_text(
            item["caixas"].replace("|", "\n"), encoding="utf-8",
        )
    data = {
        "path": str(destino.resolve()), **{s: f"images/{s}" for s in SPLITS},
        "names": dict(enumerate(CLASSES)),
    }
    (destino / "data.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifesto", type=Path)
    parser.add_argument("bases", type=Path)
    parser.add_argument("destino", type=Path)
    args = parser.parse_args()
    preparar(args.manifesto, args.bases, args.destino)
