"""
Pré-processamento: remove régua Kodak + recorta gravura pelo fundo
==================================================================
Pipeline por imagem:
  1. Detecta régua Kodak (patches cyan+yellow) → corta tudo abaixo
  2. Amostra a cor do fundo nos 4 cantos da imagem
  3. Varre linhas e colunas: onde > CONTENT_FRAC dos pixels difere
     do fundo → é borda da gravura → recorta com BBOX_PADDING de margem

Percorre imagens/<ID>/, pega a primeira imagem de cada pasta e salva
em imagens_crop/<ID>.jpg (pasta flat).

Execute UMA VEZ antes do cluster_portraits.py.
Depois aponte IMAGE_DIR = "./imagens_crop" no cluster_portraits.py.
"""

import numpy as np
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── CONFIG ────────────────────────────────────────────────────────────────────

INPUT_DIR  = r"D:\HERMES\images"
OUTPUT_DIR = "./imagens_crop"

# Kodak
KODAK_SEARCH_FRAC = 0.55  # busca a régua só nos últimos X% da imagem
KODAK_MARGIN_PX   = 30    # pixels de margem acima da régua

# Detecção do fundo
BG_CORNER_PX  = 80    # tamanho do quadrado amostrado em cada canto para estimar o fundo
BG_TOLERANCE  = 18    # diferença de brilho pra pixel ser considerado "não-fundo"
CONTENT_FRAC  = 0.03  # fração mínima de pixels não-fundo por linha/coluna (3%)
BBOX_PADDING  = 80    # pixels de margem ao redor da gravura detectada

MAX_WORKERS    = 8
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# ── ETAPA 1: KODAK ────────────────────────────────────────────────────────────

def remove_kodak(arr: np.ndarray) -> tuple[np.ndarray, str]:
    h = arr.shape[0]
    search_from = int(h * KODAK_SEARCH_FRAC)
    r = arr[search_from:, :, 0].astype(np.int16)
    g = arr[search_from:, :, 1].astype(np.int16)
    b = arr[search_from:, :, 2].astype(np.int16)
    cyan   = ((r < 120) & (g > 140) & (b > 140)).sum(axis=1)
    yellow = ((r > 140) & (g > 140) & (b < 120)).sum(axis=1)
    hits = np.where((cyan > 50) & (yellow > 50))[0]
    if len(hits) > 0:
        cut = max(0, search_from + int(hits[0]) - KODAK_MARGIN_PX)
        return arr[:cut], f"kodak@{search_from + int(hits[0])}"
    return arr, "sem-kodak"

# ── ETAPA 2: RECORTE PELO FUNDO ───────────────────────────────────────────────

def crop_by_background(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    p = BG_CORNER_PX

    # Amostra os 4 cantos para estimar o fundo
    corners = [
        arr[:p,    :p   ],
        arr[:p,    w-p: ],
        arr[h-p:,  :p   ],
        arr[h-p:,  w-p: ],
    ]
    bg_mean = float(np.vstack([c.reshape(-1, 3) for c in corners]).mean())

    gray = np.mean(arr, axis=2)
    not_bg_row = (np.abs(gray - bg_mean) >= BG_TOLERANCE).sum(axis=1) / w
    not_bg_col = (np.abs(gray - bg_mean) >= BG_TOLERANCE).sum(axis=0) / h

    rows = np.where(not_bg_row > CONTENT_FRAC)[0]
    cols = np.where(not_bg_col > CONTENT_FRAC)[0]

    if len(rows) == 0 or len(cols) == 0:
        return arr  # nada detectado, devolve tudo

    t = max(0, rows[0]  - BBOX_PADDING)
    b = min(h, rows[-1] + BBOX_PADDING)
    l = max(0, cols[0]  - BBOX_PADDING)
    r = min(w, cols[-1] + BBOX_PADDING)

    return arr[t:b, l:r]

# ── PROCESSAMENTO DE UMA IMAGEM ───────────────────────────────────────────────

def process_image(src_path: Path, dst_path: Path) -> str:
    if dst_path.exists():
        return f"  skip  {src_path.parent.name}"
    try:
        arr = np.array(Image.open(src_path).convert("RGB"))

        arr, kodak_note = remove_kodak(arr)
        arr = crop_by_background(arr)

        fh, fw = arr.shape[:2]
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(dst_path, quality=95)
        return f"  ok    {src_path.parent.name}  ({kodak_note} | {fw}x{fh}px)"
    except Exception as e:
        return f"  ERR   {src_path.parent.name}  — {e}"

# ── COLETA DE ARQUIVOS ────────────────────────────────────────────────────────

def collect_pairs(input_dir: str, output_dir: str) -> list[tuple[Path, Path]]:
    root_in  = Path(input_dir)
    root_out = Path(output_dir)
    pairs    = []
    subdirs  = sorted([d for d in root_in.iterdir() if d.is_dir()])
    if subdirs:
        for subdir in subdirs:
            imgs = sorted([p for p in subdir.iterdir()
                           if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])
            if imgs:
                src = imgs[0]
                pairs.append((src, root_out / f"{subdir.name}{src.suffix.lower()}"))
            else:
                print(f"  ⚠  sem imagem em: {subdir}")
    else:
        for src in sorted([p for p in root_in.iterdir()
                           if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]):
            pairs.append((src, root_out / src.name))
    return pairs

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\nBuscando imagens em '{INPUT_DIR}' …")
    pairs = collect_pairs(INPUT_DIR, OUTPUT_DIR)
    print(f"  {len(pairs)} imagens encontradas\n")

    total = len(pairs)
    ok = skipped = errors = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_image, src, dst): (src, dst)
                   for src, dst in pairs}
        for i, future in enumerate(as_completed(futures), 1):
            msg = future.result()
            print(f"[{i:>4}/{total}] {msg}")
            if   "skip" in msg: skipped += 1
            elif "ERR"  in msg: errors  += 1
            else:               ok      += 1

    print(f"""
─────────────────────────────
  Total        : {total}
  Processadas  : {ok}
  Já existiam  : {skipped}
  Erros        : {errors}
─────────────────────────────
Imagens salvas em '{OUTPUT_DIR}/'

Próximo passo → no cluster_portraits.py altere:
  IMAGE_DIR = "./imagens_crop"
""")

if __name__ == "__main__":
    main()