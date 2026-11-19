"""Amplia o conjunto de treino com fotos de celular de Roma e ajusta o YOLO."""
import argparse
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import re
from shutil import copytree
from zipfile import ZipFile

from PIL import Image
import yaml

from .preparar_publico import CLASSES

FORMATOS = {".jpg", ".jpeg", ".png", ".webp"}


def data_captura(nome: str) -> str | None:
    data = re.search(r"(20\d{2})-?(\d{2})-?(\d{2})[_-]", Path(nome).name)
    return "".join(data.groups()) if data else None


def caixas_buraco(texto: str) -> list[str]:
    caixas = []
    for linha in texto.splitlines():
        campos = linha.split()
        if not campos or campos[0] != "0":
            continue
        coordenadas = list(map(float, campos[1:]))
        if len(coordenadas) == 4:
            cx, cy, largura, altura = coordenadas
            x1, x2 = cx - largura / 2, cx + largura / 2
            y1, y2 = cy - altura / 2, cy + altura / 2
        elif len(coordenadas) >= 6 and len(coordenadas) % 2 == 0:
            xs, ys = coordenadas[::2], coordenadas[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        else:
            raise ValueError("Formato de anotação de buraco inválido.")
        x1, x2, y1, y2 = max(0.0, x1), min(1.0, x2), max(0.0, y1), min(1.0, y2)
        if x2 > x1 and y2 > y1:
            caixas.append(f"0 {(x1+x2)/2:.7f} {(y1+y2)/2:.7f} {x2-x1:.7f} {y2-y1:.7f}")
    return caixas


def hash_visual(dados: bytes) -> int:
    with Image.open(io.BytesIO(dados)) as foto:
        pixels = list(foto.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    return sum(
        (pixels[i] > pixels[i + 1]) << (i - i // 9)
        for i in range(72) if i % 9 < 8
    )


def acrescentar(base: Path, arquivo: Path, manifesto: Path, destino: Path) -> dict:
    with manifesto.open(encoding="utf-8-sig", newline="") as f:
        linhas = list(csv.DictReader(f))
    rome = [x for x in linhas if x["fonte"] == "rome"]
    datas_reservadas = {data_captura(x["origem"]) for x in rome if x["split"] in {"val", "test"}}
    hashes = {x["sha256"] for x in linhas}
    reservados = [
        hash_visual(p.read_bytes()) for split in ("val", "test")
        for p in (base / "images" / split).iterdir()
        if p.suffix.lower() in FORMATOS
    ]
    config = yaml.safe_load((base / "data.yaml").read_text(encoding="utf-8"))
    if config.get("names") != dict(enumerate(CLASSES)):
        raise ValueError("O conjunto e o modelo devem seguir os IDs do Calçada Segura.")
    copytree(base, destino, ignore=lambda _, nomes: [n for n in nomes if n.endswith(".cache")])
    config["path"] = str(destino.resolve())
    (destino / "data.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    novos = []
    with ZipFile(arquivo) as pacote:
        fotos = sorted(n for n in pacote.namelist() if "/images/" in n
                       and re.fullmatch(r"20\d{6}_\d{6}\.jpg", Path(n).name))
        for nome in fotos:
            if data_captura(nome) in datas_reservadas:
                continue
            binario = pacote.read(nome)
            sha = sha256(binario).hexdigest()
            if sha in hashes:
                continue
            visual = hash_visual(binario)
            if any((visual ^ outro).bit_count() <= 3 for outro in reservados):
                continue
            rotulo = nome.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            caixas = caixas_buraco(pacote.read(rotulo).decode("utf-8-sig"))
            foto = "roma-celular-" + Path(nome).name
            (destino / "images/train" / foto).write_bytes(binario)
            (destino / "labels/train" / Path(foto).with_suffix(".txt")).write_text(
                "\n".join(caixas), encoding="utf-8")
            novos.append({"imagem": foto, "origem": nome, "sha256": sha, "caixas": len(caixas)})
            hashes.add(sha)
            reservados.append(visual)
    if not any(x["caixas"] for x in novos):
        raise ValueError("O reforço não contém buracos anotados.")
    dados = {
        "fonte": "https://doi.org/10.5281/zenodo.17834373",
        "dados_exclusivos_de_treino": novos,
        "imagens_com_buraco": sum(x["caixas"] > 0 for x in novos),
        "imagens_sem_buraco": sum(x["caixas"] == 0 for x in novos),
        "caixas_buraco": sum(x["caixas"] for x in novos),
    }
    (destino / "reforco-roma.json").write_text(json.dumps(dados, indent=2), encoding="utf-8")
    return dados


def ajustar(pesos: Path, dados: Path, saida: Path, device: str, epocas: int, imgsz: int, batch: int, lr: float):
    from ultralytics import YOLO

    if saida.exists():
        raise ValueError("Informe uma pasta nova para manter as execuções anteriores.")
    YOLO(str(pesos)).train(
        data=str(dados.resolve()), device=device, imgsz=imgsz, batch=batch, epochs=epocas,
        workers=0, seed=42, optimizer="AdamW", lr0=lr, lrf=0.2, warmup_epochs=1,
        mosaic=0.10, close_mosaic=5, mixup=0.0, scale=0.20, translate=0.05, freeze=7, patience=8,
        project=str(saida.parent.resolve()), name=saida.name, plots=True,
    )
    return saida / "weights/best.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--roma", type=Path, required=True)
    parser.add_argument("--manifesto", type=Path, required=True)
    parser.add_argument("--dados-ampliados", type=Path, required=True)
    parser.add_argument("--pesos", type=Path, required=True)
    parser.add_argument("--saida", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epocas", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=448)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.00055)
    args = parser.parse_args()
    acrescentar(args.base, args.roma, args.manifesto, args.dados_ampliados)
    ajustar(args.pesos, args.dados_ampliados / "data.yaml", args.saida,
            args.device, args.epocas, args.imgsz, args.batch, args.lr)
