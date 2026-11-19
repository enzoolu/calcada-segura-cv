"""Executa os dois experimentos com bases preparadas e exporta o teste separado."""
import argparse
from pathlib import Path

from ultralytics import YOLO

from .treinar import avaliar

AJUSTES = {
    "calcada": dict(imgsz=480, batch=12, epochs=14, lr0=0.00065, lrf=0.18,
                  mosaic=0, close_mosaic=0, scale=0.18, patience=8),
    "tres-classes": dict(imgsz=384, batch=16, epochs=21, lr0=0.0012, lrf=0.2,
                        mosaic=0.25, close_mosaic=7, scale=0.25, patience=9),
}


def treinar(variacao, dados, pesos, saida, device):
    ajustes = AJUSTES[variacao]
    if saida.exists():
        raise ValueError("Use uma pasta nova para preservar as evidências anteriores.")
    YOLO(str(pesos)).train(
        data=str(dados.resolve()), device=device, seed=42, workers=0,
        optimizer="AdamW", mixup=0, warmup_epochs=1, translate=0.06, freeze=7,
        project=str(saida.parent.resolve()), name=saida.name, plots=True, **ajustes,
    )
    avaliar(saida / "weights" / "best.pt", dados.resolve(), saida / "teste", ajustes["imgsz"], device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variacao", choices=AJUSTES)
    parser.add_argument("--dados", type=Path, required=True)
    parser.add_argument("--pesos", type=Path, required=True)
    parser.add_argument("--saida", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    treinar(args.variacao, args.dados, args.pesos, args.saida, args.device)
