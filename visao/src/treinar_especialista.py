"""Ajusta um YOLO11n de uma classe para cavidades visíveis na calçada."""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, required=True)
    parser.add_argument("--pesos-iniciais", type=Path,
                        default=Path("visao/evidencias/refino-buraco/best.pt"))
    parser.add_argument("--saida", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epocas", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--congelar", type=int, default=6)
    args = parser.parse_args()
    for arquivo in (args.dados, args.pesos_iniciais):
        if not arquivo.is_file():
            parser.error(f"Arquivo não encontrado: {arquivo}")
    saida = args.saida.resolve()
    if saida.exists():
        parser.error("--saida já existe. Informe outro diretório de treino.")
    if min(args.epocas, args.threads, args.imgsz) < 1 or args.congelar < 0:
        parser.error("Confira as épocas, threads, resolução e camadas congeladas.")

    torch.set_num_threads(args.threads)
    modelo = YOLO(str(args.pesos_iniciais.resolve()))
    if modelo.names != {0: "buraco"}:
        parser.error("O treino exige um peso inicial com a classe única 'buraco'.")
    modelo.train(
        data=str(args.dados.resolve()), project=str(saida.parent), name=saida.name,
        epochs=args.epocas, patience=7, imgsz=args.imgsz, batch=8, workers=0,
        device=args.device, seed=42, deterministic=True, freeze=args.congelar,
        optimizer="AdamW", lr0=0.0007, lrf=0.10, warmup_epochs=1.0,
        mosaic=0.35, close_mosaic=3, scale=0.22, degrees=5.0,
        perspective=0.0002, amp=False, plots=True,
    )


if __name__ == "__main__":
    main()
