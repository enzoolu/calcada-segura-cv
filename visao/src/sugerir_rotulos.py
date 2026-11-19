"""Gera sugestões YOLO para revisão a partir dos quadros de treino e validação."""
import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from ultralytics import YOLO

from .inferir import ler_limiares
from .preparar_publico import CLASSES

EXTENSOES = {".jpg", ".jpeg", ".png", ".webp"}


def sugerir(quadros, destino, pesos, confianca, imgsz, device, perfil=None):
    if destino.exists():
        raise ValueError("Escolha uma pasta nova para as sugestões.")
    modelo = YOLO(str(pesos))
    nomes = modelo.names
    if not nomes or any(nome not in CLASSES for nome in nomes.values()):
        raise ValueError("Use um modelo treinado para as classes do Calçada Segura.")
    limiares = ler_limiares(perfil, nomes, confianca)
    arquivos = {
        split: sorted(
            p for p in (quadros / split / "images").iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSOES
        )
        for split in ("train", "valid")
    }
    if not any(arquivos.values()):
        raise ValueError("A extração precisa conter imagens de treino ou validação.")
    destino.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destino.parent) as temporario:
        saida = Path(temporario) / "sugestoes"
        saida.mkdir()
        with (saida / "revisao.csv").open("w", encoding="utf-8-sig", newline="") as arq:
            tabela = csv.writer(arq)
            tabela.writerow(["split", "imagem", "classe", "confianca", "cx", "cy", "w", "h"])
            for split, imagens in arquivos.items():
                rotulos = saida / split / "labels"
                rotulos.mkdir(parents=True)
                if len({imagem.stem.casefold() for imagem in imagens}) != len(imagens):
                    raise ValueError(f"Imagens com o mesmo nome base em {split}.")
                if not imagens:
                    continue
                resultados = modelo.predict(
                    [str(p) for p in imagens], stream=True, batch=8, conf=min(limiares.values()),
                    imgsz=imgsz, device=device, verbose=False,
                )
                for imagem, resultado in zip(imagens, resultados, strict=True):
                    linhas = []
                    for classe, caixas, score in zip(
                        resultado.boxes.cls.tolist(), resultado.boxes.xywhn.tolist(),
                        resultado.boxes.conf.tolist(), strict=True,
                    ):
                        nome = nomes[int(classe)]
                        if score < limiares[nome]:
                            continue
                        formas = [f"{x:.6f}" for x in caixas]
                        tabela.writerow([split, imagem.name, nome, f"{score:.4f}", *formas])
                        linhas.append(" ".join([str(CLASSES.index(nome)), *formas]))
                    if linhas:
                        (rotulos / f"{imagem.stem}.txt").write_text("\n".join(linhas), encoding="utf-8")
                    else:
                        tabela.writerow([split, imagem.name, "", "", "", "", "", ""])
        if destino.exists():
            raise ValueError("A pasta de saída já existe.")
        saida.rename(destino)
    print(f"{sum(map(len, arquivos.values()))} imagens na fila de revisão.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quadros", type=Path)
    parser.add_argument("destino", type=Path)
    parser.add_argument("--pesos", type=Path, required=True)
    parser.add_argument("--confianca", type=float, default=0.35)
    parser.add_argument("--perfil", type=Path, help="JSON opcional com limiares por classe")
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not 0 < args.confianca <= 1 or args.imgsz < 32:
        parser.error("Informe confiança em (0, 1] e imagem de pelo menos 32 px.")
    try:
        sugerir(args.quadros, args.destino, args.pesos, args.confianca, args.imgsz, args.device, args.perfil)
    except (ValueError, OSError) as erro:
        parser.exit(2, f"{erro}\n")
