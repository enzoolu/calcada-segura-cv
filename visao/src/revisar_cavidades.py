"""Aplica a revisão de cavidades em Pune sem reescrever as partições originais de avaliação."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from shutil import copy2, copytree

import yaml

from .calibrar import caixa_xyxy, intersecao_sobre_uniao


def linhas_buraco(caixas):
    linhas = []
    for x1, y1, x2, y2 in caixas:
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("Caixa inválida no manifesto de cavidades.")
        linhas.append("0 " + " ".join(f"{valor:.7f}" for valor in (
            (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1,
        )))
    return linhas


def preparar(base, manifesto, treino, checagem):
    dados = json.loads(manifesto.read_text(encoding="utf-8"))
    for item in dados["imagens"]:
        imagem = item["imagem"]
        split = item["split"]
        if Path(imagem).name != imagem or split not in {"train", "val", "test"}:
            raise ValueError("Imagem ou divisão inválida no manifesto de cavidades.")
        caminho = base / "images" / split / imagem
        if sha256(caminho.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Arquivo divergente do manifesto: {caminho}")
    copytree(base, treino, ignore=lambda _, nomes: [n for n in nomes if n.endswith(".cache")])
    config = yaml.safe_load((base/"data.yaml").read_text(encoding="utf-8"))
    config["path"] = str(treino.resolve())
    (treino/"data.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    checagem.mkdir(parents=True, exist_ok=False)
    for split in ("val","test"):
        for grupo in ("images","labels"):
            (checagem/grupo/split).mkdir(parents=True)
    for item in dados["imagens"]:
        split = item["split"]
        imagem = item["imagem"]
        rotulo = Path(imagem).with_suffix(".txt").name
        caixas = item["caixas_xyxy"]
        novas = linhas_buraco(caixas)
        origem = base/"images"/split/imagem
        if split == "train":
            lab = treino/"labels/train"/rotulo
            for linha in lab.read_text(encoding="utf-8").splitlines():
                if not linha.strip():
                    continue
                classe, *coordenadas = linha.split()
                antiga = caixa_xyxy(tuple(map(float, coordenadas)))
                if classe != "1" or all(intersecao_sobre_uniao(antiga, box) < 0.35 for box in caixas):
                    novas.append(linha)
            conteudo = "\n".join(novas) + "\n"
            lab.write_text(conteudo, encoding="utf-8")
            copy2(origem, treino/"images/train"/f"reforco-{imagem}")
            (treino/"labels/train"/f"reforco-{rotulo}").write_text(conteudo, encoding="utf-8")
        else:
            copy2(origem, checagem/"images"/split/imagem)
            (checagem/"labels"/split/rotulo).write_text("\n".join(novas), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--manifesto", type=Path, required=True)
    parser.add_argument("--treino", type=Path, required=True)
    parser.add_argument("--checagem", type=Path, required=True)
    parser.add_argument("--pesos", type=Path, help="Pesos iniciais para o treino opcional")
    parser.add_argument("--ajuste", type=Path, help="Pasta nova para o treino opcional")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if (args.pesos is None) != (args.ajuste is None):
        parser.error("informe --pesos e --ajuste juntos")
    preparar(args.base, args.manifesto, args.treino, args.checagem)
    if args.ajuste:
        from .refinar_buracos import ajustar

        ajustar(args.pesos, args.treino / "data.yaml", args.ajuste, args.device,
                epocas=6, imgsz=448, batch=12, lr=0.0005)
