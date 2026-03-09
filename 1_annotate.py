"""
Script 1: Ferramenta de anotação
=================================
Mostra 150 imagens aleatórias da coleção uma por uma.
Você desenha o bbox do retrato com o mouse e salva.
Resultado: pasta  dataset/  pronta pra fine-tune YOLO26.

Controles:
  - Arraste o mouse para desenhar o bbox
  - ENTER  → confirma e vai pra próxima
  - R      → refaz o bbox atual
  - S      → pula a imagem (sem anotar)
  - Q      → sai e salva o que já foi feito
"""

import cv2
import numpy as np
import random
import shutil
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────

IMAGE_DIR      = r"D:\HERMES\images"       # pasta com subpastas de IDs
DATASET_DIR    = "./dataset"       # onde salvar o dataset
N_IMAGES       = 55               # quantas imagens anotar
VAL_SPLIT      = 0.15              # fração de validação
CLASS_NAME     = "portrait"        # nome da classe
DISPLAY_HEIGHT = 900               # altura da janela de anotação (px)
SEED           = 43

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# ── COLETA IMAGENS ────────────────────────────────────────────────────────────

def collect_images(image_dir: str) -> list[Path]:
    root = Path(image_dir)
    paths = []
    subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if subdirs:
        for subdir in subdirs:
            imgs = sorted([p for p in subdir.iterdir()
                           if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])
            if imgs:
                paths.append(imgs[0])
    else:
        paths = sorted([p for p in root.iterdir()
                        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])
    random.seed(SEED)
    random.shuffle(paths)
    return paths[:N_IMAGES]

# ── ANOTAÇÃO COM MOUSE ────────────────────────────────────────────────────────

class Annotator:
    def __init__(self):
        self.drawing = False
        self.bbox    = None   # (x1, y1, x2, y2) em coords da imagem exibida
        self.start   = None

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start   = (x, y)
            self.bbox    = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.bbox = (self.start[0], self.start[1], x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.bbox    = (self.start[0], self.start[1], x, y)

    def annotate(self, img_path: Path) -> tuple | None:
        """
        Retorna (x1,y1,x2,y2) normalizadas [0-1] no tamanho original,
        ou None se pulado.
        """
        orig = cv2.imdecode(
            np.frombuffer(img_path.read_bytes(), np.uint8),
            cv2.IMREAD_COLOR
        )
        if orig is None:
            print(f"  Erro ao abrir {img_path}")
            return None

        oh, ow = orig.shape[:2]
        scale  = DISPLAY_HEIGHT / oh
        dw, dh = int(ow * scale), DISPLAY_HEIGHT
        disp   = cv2.resize(orig, (dw, dh))

        self.bbox    = None
        self.drawing = False

        win = "Anotação — arraste bbox | ENTER=ok  R=refaz  S=pula  Q=sair"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, dw, dh)
        cv2.setMouseCallback(win, self.mouse_cb)

        while True:
            frame = disp.copy()
            if self.bbox:
                x1, y1, x2, y2 = self.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, img_path.parent.name, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            cv2.imshow(win, frame)
            key = cv2.waitKey(20) & 0xFF

            if key == 13 and self.bbox:   # ENTER
                cv2.destroyWindow(win)
                x1, y1, x2, y2 = [min(a, b) for a, b in
                                   zip(self.bbox[:2], self.bbox[2:4])] + \
                                  [max(a, b) for a, b in
                                   zip(self.bbox[:2], self.bbox[2:4])]
                # normaliza de volta pra original
                x1n = (x1 / dw); y1n = (y1 / dh)
                x2n = (x2 / dw); y2n = (y2 / dh)
                cx  = (x1n + x2n) / 2
                cy  = (y1n + y2n) / 2
                bw  = x2n - x1n
                bh  = y2n - y1n
                return cx, cy, bw, bh

            elif key == ord('r'):         # refaz
                self.bbox = None
            elif key == ord('s'):         # pula
                cv2.destroyWindow(win)
                return None
            elif key == ord('q'):         # sai
                cv2.destroyWindow(win)
                return "quit"

# ── SALVA DATASET ─────────────────────────────────────────────────────────────

def save_label(label_path: Path, cx, cy, bw, bh):
    label_path.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

def write_yaml(dataset_dir: Path, class_name: str):
    yaml = dataset_dir / "data.yaml"
    yaml.write_text(
        f"path: {dataset_dir.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"nc: 1\n"
        f"names: ['{class_name}']\n"
    )
    print(f"  YAML salvo em {yaml}")

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    images = collect_images(IMAGE_DIR)
    print(f"  {len(images)} imagens selecionadas para anotação\n")
    print("  Controles: arraste bbox → ENTER confirma | R refaz | S pula | Q sai\n")

    ds   = Path(DATASET_DIR)
    ann  = Annotator()
    annotated: list[tuple[Path, tuple]] = []

    for i, img_path in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] {img_path.parent.name}", end="  ", flush=True)
        result = ann.annotate(img_path)
        if result == "quit":
            print("→ saindo")
            break
        elif result is None:
            print("→ pulado")
        else:
            annotated.append((img_path, result))
            print(f"→ anotado  cx={result[0]:.3f} cy={result[1]:.3f}")

    if not annotated:
        print("Nenhuma imagem anotada.")
        return

    # Split train/val
    random.shuffle(annotated)
    n_val   = max(1, int(len(annotated) * VAL_SPLIT))
    val_set = annotated[:n_val]
    trn_set = annotated[n_val:]

    for split, subset in [("train", trn_set), ("val", val_set)]:
        img_dir = ds / "images" / split
        lbl_dir = ds / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, (cx, cy, bw, bh) in subset:
            stem = img_path.parent.name          # ID da imagem como nome
            dst_img = img_dir / f"{stem}{img_path.suffix.lower()}"
            dst_lbl = lbl_dir / f"{stem}.txt"
            shutil.copy2(img_path, dst_img)
            save_label(dst_lbl, cx, cy, bw, bh)

    write_yaml(ds, CLASS_NAME)

    print(f"""
─────────────────────────────
  Anotadas : {len(annotated)}
  Train    : {len(trn_set)}
  Val      : {len(val_set)}
─────────────────────────────
Dataset salvo em '{DATASET_DIR}/'
Próximo passo → python 2_finetune.py
""")

if __name__ == "__main__":
    main()
