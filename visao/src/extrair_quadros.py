"""Extrai imagens de vídeos por rota e prepara o inventário para anotação."""
import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

EXTENSOES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
DIVISOES = {"treino": "train", "validacao": "valid", "teste": "test"}


def extrair(video, destino, intervalo):
    captura = cv2.VideoCapture(str(video))
    try:
        fps = captura.get(cv2.CAP_PROP_FPS)
        if not captura.isOpened() or not 0 < fps < 1000:
            raise ValueError(f"Vídeo sem uma taxa de quadros legível: {video.name}")
        passo = max(1, round(fps * intervalo))
        total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError(f"Duração de vídeo indisponível: {video.name}")
        for indice in range(0, total, passo):
            captura.set(cv2.CAP_PROP_POS_FRAMES, indice)
            ok, quadro = captura.read()
            if not ok:
                raise ValueError(f"Falha ao ler {video.name} perto de {indice / fps:.2f}s")
            nome = f"{video.stem}_{indice:07d}.jpg"
            if not cv2.imwrite(str(destino / nome), quadro, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise OSError(f"Não foi possível salvar {nome}")
            yield nome, round(indice / fps, 3)
    finally:
        captura.release()


def preparar(origem, destino, intervalo):
    if destino.exists():
        raise ValueError("Escolha uma pasta de saída nova.")
    lotes = {}
    nomes = set()
    for pasta, split in DIVISOES.items():
        local = origem / pasta
        videos = sorted(p for p in local.iterdir() if p.is_file() and p.suffix.lower() in EXTENSOES)
        if not videos:
            raise ValueError(f"Adicione pelo menos um vídeo em {local}.")
        for video in videos:
            if video.stem.casefold() in nomes:
                raise ValueError(f"Dê nomes diferentes aos vídeos de cada rota: {video.name}.")
            nomes.add(video.stem.casefold())
        lotes[split] = videos
    destino.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destino.parent) as temporario:
        pasta = Path(temporario) / "quadros"
        pasta.mkdir()
        with (pasta / "inventario.csv").open("w", encoding="utf-8-sig", newline="") as arquivo:
            tabela = csv.writer(arquivo)
            tabela.writerow(["split", "video", "segundos", "imagem"])
            for split, videos in lotes.items():
                imagens = pasta / split / "images"
                imagens.mkdir(parents=True)
                (pasta / split / "labels").mkdir()
                for video in videos:
                    for nome, segundo in extrair(video, imagens, intervalo):
                        tabela.writerow([split, video.name, segundo, nome])
        if destino.exists():
            raise ValueError("A pasta de saída já foi criada por outra execução.")
        pasta.rename(destino)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("origem", type=Path, help="pasta com treino/, validacao/ e teste/")
    parser.add_argument("destino", type=Path)
    parser.add_argument("--intervalo", type=float, default=2, help="segundos entre quadros")
    args = parser.parse_args()
    if not 0 < args.intervalo < 600:
        parser.error("Use um intervalo maior que zero e menor que 600 segundos.")
    try:
        preparar(args.origem, args.destino, args.intervalo)
    except (OSError, ValueError, cv2.error) as erro:
        parser.exit(2, f"{erro}\n")


if __name__ == "__main__":
    main()
