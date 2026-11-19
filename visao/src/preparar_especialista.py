"""Reconstrói o conjunto de cavidades a partir dos manifestos e acervos públicos."""
import argparse
from contextlib import nullcontext
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import random
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from PIL import Image, ImageOps

PASTA_PUNE = "Final Indian Footpath Damage Segmentation Dataset1"


def linha_yolo(caixa):
    x1, y1, x2, y2 = caixa
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"Caixa inválida: {caixa}")
    valores = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    return "0 " + " ".join(f"{v:.7f}" for v in valores)


def escrever(pasta, split, nome, jpeg, rotulos):
    imagem = pasta / "images" / split / nome
    anotacao = pasta / "labels" / split / imagem.with_suffix(".txt").name
    imagem.parent.mkdir(parents=True, exist_ok=True)
    anotacao.parent.mkdir(parents=True, exist_ok=True)
    imagem.write_bytes(jpeg)
    anotacao.write_text("".join(linha + "\n" for linha in rotulos), encoding="utf-8")


def preparar_pune(arquivo):
    with Image.open(BytesIO(arquivo)) as original:
        foto = ImageOps.exif_transpose(original).convert("RGB")
    foto.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    return para_jpeg(foto, qualidade=94)


def para_jpeg(imagem, qualidade):
    destino = BytesIO()
    imagem.save(destino, format="JPEG", quality=qualidade)
    return destino.getvalue()


def recorte_de_treino(imagem, caixa, rng):
    largura, altura = imagem.size
    x1, y1, x2, y2 = (p * tamanho for p, tamanho in zip(caixa, (largura, altura) * 2, strict=True))
    centro_x, centro_y = (x1 + x2) / 2, (y1 + y2) / 2
    escala = rng.uniform(0.49, 0.72)
    w = min(largura, max(round(escala * largura), round(2.15 * (x2 - x1))))
    h = min(altura, max(round(escala * altura), round(2.15 * (y2 - y1))))
    if w == largura and h == altura:
        return None
    x = max(0, min(largura - w, round(centro_x - w / 2)))
    y = max(0, min(altura - h, round(centro_y - h / 2)))
    relativa = ((x1 - x) / w, (y1 - y) / h, (x2 - x) / w, (y2 - y) / h)
    return imagem.crop((x, y, x + w, y + h)), relativa


def preparar(fonte, destino, pune_zip, roma_zip, imagens_pune, recortar):
    registros = json.loads(fonte.read_text(encoding="utf-8"))
    rng = random.Random(42)
    recortes, hashes_pune_diferentes = 0, []
    fonte_pune = ZipFile(pune_zip) if pune_zip else nullcontext()
    with ZipFile(roma_zip) as roma, fonte_pune as pune:
        for item in registros:
            split, nome = item["split"], item["imagem"]
            if item["fonte"] == "roma":
                jpeg = roma.read(item["origem"])
                if sha256(jpeg).hexdigest() != item["sha256"]:
                    raise ValueError(f"Arquivo de Roma difere da referência: {nome}")
                escrever(destino, split, nome, jpeg, item["caixas_yolo"])
                continue
            jpeg = ((imagens_pune / item["origem"]).read_bytes() if imagens_pune
                    else preparar_pune(pune.read(f"{PASTA_PUNE}/{item['origem']}")))
            if sha256(jpeg).hexdigest() != item["sha256"]:
                hashes_pune_diferentes.append(nome)
            caixas = item["caixas_xyxy"]
            escrever(destino, split, nome, jpeg, [linha_yolo(b) for b in caixas])
            if recortar and split == "train" and len(caixas) == 1:
                with Image.open(BytesIO(jpeg)) as imagem:
                    recorte = recorte_de_treino(imagem, caixas[0], rng)
                if recorte:
                    foto, caixa = recorte
                    escrever(destino, split, "recorte-" + nome, para_jpeg(foto, 92), [linha_yolo(caixa)])
                    recortes += 1
    return recortes, hashes_pune_diferentes


def preparar_sp(referencia, pasta_videos, destino):
    fontes = json.loads(referencia.read_text(encoding="utf-8"))
    with TemporaryDirectory() as temporario:
        for ponto in fontes["pontos"]:
            video = pasta_videos / f"{ponto['id']}.mov"
            if sha256(video.read_bytes()).hexdigest() != ponto["sha256"]:
                raise ValueError(f"O vídeo histórico difere da referência: {video.name}")
            pasta = Path(temporario) / ponto["id"]
            pasta.mkdir()
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video), "-map", "0:v:0",
                "-vf", fontes["filtro_ffmpeg"], "-q:v", "3", str(pasta / "%02d.jpg"),
            ], check=True)
            for quadro in ponto["quadros_anotados"]:
                jpeg = (pasta / f'{quadro["numero"]:02}.jpg').read_bytes()
                escrever(destino, "sp", quadro["imagem"], jpeg, [linha_yolo(b) for b in quadro["caixas_xyxy"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifesto", type=Path, required=True)
    origem = parser.add_mutually_exclusive_group(required=True)
    origem.add_argument("--pune-zip", type=Path, help="ZIP oficial de Pune")
    origem.add_argument("--imagens-pune", type=Path, help="Pasta com JPEGs pré-processados em Train, Val e Test")
    parser.add_argument("--roma-zip", type=Path, required=True)
    parser.add_argument("--referencia-sp", type=Path)
    parser.add_argument("--videos-sp", type=Path, help="Pasta com as versões transcodificadas indicadas no manifesto")
    parser.add_argument("--recortar-treino", action="store_true")
    parser.add_argument("--saida", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.referencia_sp) != bool(args.videos_sp):
        parser.error("Informe --referencia-sp e --videos-sp juntos.")
    destino = args.saida.resolve()
    destino.mkdir(exist_ok=False, parents=True)
    recortes, diferentes = preparar(
        args.manifesto, destino, args.pune_zip, args.roma_zip, args.imagens_pune, args.recortar_treino,
    )
    if args.referencia_sp:
        preparar_sp(args.referencia_sp, args.videos_sp, destino)
    # JSON é um subconjunto de YAML e preserva caminhos Windows e POSIX.
    dados = {"path": str(destino), "train": "images/train", "val": "images/val", "test": "images/test",
             "names": ["buraco"]}
    (destino / "data.yaml").write_text(json.dumps(dados, indent=2), encoding="utf-8")
    print(json.dumps({"recortes_de_treino": recortes, "pune_com_jpeg_diferente": diferentes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
