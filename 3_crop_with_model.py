"""
Script 3: Recorta todas as imagens com o modelo fine-tunado
============================================================
Usa o YOLO26s fine-tunado pra detectar o retrato e recortar
com margem. Salva em imagens_crop/<ID>.jpg (pasta flat).

Fallback (quando modelo não detecta):
  1. Remove régua Kodak (cyan+yellow)
  2. Recorta pelo fundo (bg color)

Uso:
  python 3_crop_with_model.py
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO


# ── CONFIG ────────────────────────────────────────────────────────────────────

INPUT_DIR    = "D:\HERMES\images"
OUTPUT_DIR   = "D:\HERMES\imagens_crop"
MODEL_PATH   = "runs/detect/portrait_detector/weights/best.pt"
FALLBACK_LOG = "fallback_ids.txt"   # IDs que caíram no fallback
"""
Script 3: Recorta todas as imagens com o modelo fine-tunado
============================================================
Fallback em cascata quando o modelo não detecta:
  1. Contorno (Otsu + dilate) — detecta a gravura pela borda escura
  2. Kodak + bg color         — fallback final

Uso:
  python 3_crop_with_model.py
"""


CONF_THRESH  = 0.20
BBOX_PADDING = 80

# Fallback 1 — contorno
CONTOUR_SCALE       = 4      # trabalha em 1/4 da resolução
CONTOUR_BLUR        = 21     # tamanho do blur gaussiano (ímpar)
CONTOUR_DILATE      = 15     # tamanho do kernel de dilatação
CONTOUR_MIN_AREA    = 0.04   # fração mínima da imagem
CONTOUR_ASPECT_MIN  = 0.3    # aspect ratio mínimo (largura/altura)
CONTOUR_ASPECT_MAX  = 1.3    # aspect ratio máximo
CONTOUR_PADDING     = 80

# Fallback 2 — Kodak + bg color
KODAK_SEARCH_FRAC   = 0.55
KODAK_MARGIN_PX     = 30
CYAN_THRESH         = 50
YELLOW_THRESH       = 50
BG_CORNER_PX        = 80
BG_TOLERANCE        = 18
CONTENT_FRAC        = 0.03
BG_PADDING          = 80

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# ── FALLBACK 1: CONTORNO ──────────────────────────────────────────────────────

def crop_by_contour(arr: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Detecta a gravura pela borda escura usando Otsu + morfologia.
    Retorna (arr_recortado, sucesso).
    """
    h, w = arr.shape[:2]
    s = CONTOUR_SCALE
    gray  = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (w // s, h // s))
    sh, sw = small.shape

    blur = cv2.GaussianBlur(small, (CONTOUR_BLUR, CONTOUR_BLUR), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel  = np.ones((CONTOUR_DILATE, CONTOUR_DILATE), np.uint8)
    dilated = cv2.dilate(thresh, kernel)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = sh * sw * CONTOUR_MIN_AREA
    best = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / ch
        if CONTOUR_ASPECT_MIN < aspect < CONTOUR_ASPECT_MAX:
            if best is None or area > best[0]:
                best = (area, x, y, cw, ch)

    if best:
        _, x, y, cw, ch = best
        x, y, cw, ch = x*s, y*s, cw*s, ch*s
        p = CONTOUR_PADDING
        x1 = max(0, x - p);  y1 = max(0, y - p)
        x2 = min(w, x+cw+p); y2 = min(h, y+ch+p)
        return arr[y1:y2, x1:x2], True

    return arr, False


# ── FALLBACK 2: KODAK + BG COLOR ─────────────────────────────────────────────

def remove_kodak(arr: np.ndarray) -> np.ndarray:
    h = arr.shape[0]
    sf = int(h * KODAK_SEARCH_FRAC)
    r = arr[sf:, :, 0].astype(np.int16)
    g = arr[sf:, :, 1].astype(np.int16)
    b = arr[sf:, :, 2].astype(np.int16)
    cyan_c   = ((r < 120) & (g > 140) & (b > 140)).sum(axis=1)
    yellow_c = ((r > 140) & (g > 140) & (b < 120)).sum(axis=1)
    hits = np.where((cyan_c > CYAN_THRESH) & (yellow_c > YELLOW_THRESH))[0]
    if len(hits):
        cut = max(0, sf + int(hits[0]) - KODAK_MARGIN_PX)
        return arr[:cut]
    return arr


def crop_by_background(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    p = BG_CORNER_PX
    corners = [arr[:p,:p], arr[:p,w-p:], arr[h-p:,:p], arr[h-p:,w-p:]]
    bg_mean = float(np.vstack([c.reshape(-1,3) for c in corners]).mean())
    gray = np.mean(arr, axis=2)
    not_bg_row = (np.abs(gray - bg_mean) >= BG_TOLERANCE).sum(axis=1) / w
    not_bg_col = (np.abs(gray - bg_mean) >= BG_TOLERANCE).sum(axis=0) / h
    rows = np.where(not_bg_row > CONTENT_FRAC)[0]
    cols = np.where(not_bg_col > CONTENT_FRAC)[0]
    if len(rows) == 0 or len(cols) == 0:
        return arr
    t = max(0, rows[0]  - BG_PADDING); b = min(h, rows[-1] + BG_PADDING)
    l = max(0, cols[0]  - BG_PADDING); r = min(w, cols[-1] + BG_PADDING)
    return arr[t:b, l:r]


def fallback_crop(arr: np.ndarray) -> tuple[np.ndarray, str]:
    # 1. tenta contorno
    result, ok = crop_by_contour(arr)
    if ok:
        return result, "fallback-contour"

    # 2. Kodak + bg color
    result = crop_by_background(remove_kodak(arr))
    return result, "fallback-bg"


# ── COLETA IMAGENS ────────────────────────────────────────────────────────────

def collect_pairs():
    root_in  = Path(INPUT_DIR)
    root_out = Path(OUTPUT_DIR)
    pairs    = []
    subdirs  = sorted([d for d in root_in.iterdir() if d.is_dir()])
    if subdirs:
        for subdir in subdirs:
            imgs = sorted([p for p in subdir.iterdir()
                           if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])
            if imgs:
                pairs.append((imgs[0], root_out / f"{subdir.name}{imgs[0].suffix.lower()}"))
    else:
        for src in sorted([p for p in root_in.iterdir()
                           if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]):
            pairs.append((src, root_out / src.name))
    return pairs


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    model = YOLO(MODEL_PATH)
    pairs = collect_pairs()
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    total = len(pairs)
    ok = skipped = n_fallback_contour = n_fallback_bg = errors = 0
    fallback_ids = []

    for i, (src, dst) in enumerate(pairs, 1):
        if dst.exists():
            skipped += 1
            continue
        try:
            img  = Image.open(src).convert("RGB")
            w, h = img.size

            results = model.predict(img, conf=CONF_THRESH, verbose=False)
            boxes   = results[0].boxes

            if boxes and len(boxes) > 0:
                best = int(boxes.conf.cpu().numpy().argmax())
                x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy().astype(int)
                x1 = max(0, x1-BBOX_PADDING); y1 = max(0, y1-BBOX_PADDING)
                x2 = min(w, x2+BBOX_PADDING); y2 = min(h, y2+BBOX_PADDING)
                arr    = np.array(img)[y1:y2, x1:x2]
                method = f"model ({boxes.conf[best]:.2f})"
                ok += 1
            else:
                arr, method = fallback_crop(np.array(img))
                fallback_ids.append(src.parent.name)
                if "contour" in method: n_fallback_contour += 1
                else:                   n_fallback_bg      += 1

            Image.fromarray(arr).save(dst, quality=95)
            fh, fw = arr.shape[:2]
            print(f"[{i:>4}/{total}]  {src.parent.name}  {method}  {fw}x{fh}px")

        except Exception as e:
            errors += 1
            print(f"[{i:>4}/{total}]  ERR {src.parent.name}  — {e}")

    if fallback_ids:
        with open(FALLBACK_LOG, "w") as f:
            f.write("\n".join(fallback_ids) + "\n")
        print(f"\n  IDs do fallback salvos em '{FALLBACK_LOG}'")

    print(f"""
─────────────────────────────
  Total             : {total}
  Modelo            : {ok}
  Fallback contorno : {n_fallback_contour}
  Fallback bg color : {n_fallback_bg}
  Já existiam       : {skipped}
  Erros             : {errors}
─────────────────────────────
Imagens salvas em '{OUTPUT_DIR}/'
""")


if __name__ == "__main__":
    main()