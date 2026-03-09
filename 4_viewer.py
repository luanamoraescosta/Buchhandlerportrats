"""
Script 4: Gera viewer HTML interativo dos clusters
===================================================
Lê o cache de embeddings e labels salvos pelo cluster_portraits.py
e gera um HTML standalone com as imagens embutidas em base64.

Funcionalidades:
  - 8 imagens por cluster em linha
  - Botão "Ver mais 8" por cluster
  - Botão "Salvar PNG" por cluster

Uso:
  python 4_viewer.py
"""

import base64
import json
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO

# ── CONFIG ────────────────────────────────────────────────────────────────────

IMAGE_DIR      = r"D:\HERMES\imagens_crop" 
CACHE_FILE     = "embeddings_cache.npz"
LABELS_FILE    = "cluster_labels.npy"    # salvo pelo cluster_portraits.py
OUTPUT_HTML    = "cluster_viewer.html"

THUMB_W        = 200    # largura das thumbnails no viewer (px)
THUMB_QUALITY  = 75     # qualidade JPEG das thumbnails embutidas
IMAGES_PER_ROW = 8      # imagens mostradas inicialmente por cluster
SEED           = 42

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def find_images(directory: str) -> list[Path]:
    root = Path(directory)
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
    return paths


def img_to_b64(path: Path, width: int, quality: int) -> str:
    try:
        img = Image.open(path).convert("RGB")
        ratio = width / img.width
        h = int(img.height * ratio)
        img = img.resize((width, h), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    # Carrega imagens
    print("Carregando imagens …")
    image_paths = find_images(IMAGE_DIR)
    if not image_paths:
        raise FileNotFoundError(f"Nenhuma imagem em '{IMAGE_DIR}'")
    print(f"  {len(image_paths)} imagens encontradas")

    # Carrega labels
    if not Path(LABELS_FILE).exists():
        raise FileNotFoundError(
            f"'{LABELS_FILE}' não encontrado. "
            f"Rode cluster_portraits.py primeiro e certifique-se que ele salva os labels."
        )
    labels = np.load(LABELS_FILE)
    if len(labels) != len(image_paths):
        raise ValueError(f"Labels ({len(labels)}) ≠ imagens ({len(image_paths)})")

    # Agrupa por cluster
    unique_labels = sorted(set(labels))
    rng = np.random.default_rng(SEED)

    clusters = {}
    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        shuffled = rng.permutation(idx).tolist()
        clusters[int(lbl)] = shuffled

    # Converte imagens pra base64 (todas, lazy no JS)
    print("Convertendo imagens para base64 …")
    total = len(image_paths)
    b64_images = []
    for i, p in enumerate(image_paths, 1):
        if i % 50 == 0:
            print(f"  {i}/{total}")
        b64_images.append(img_to_b64(p, THUMB_W, THUMB_QUALITY))

    names = [p.stem for p in image_paths]

    # Monta dados JSON pra injetar no HTML
    data = {
        "clusters": clusters,
        "images":   b64_images,
        "names":    names,
        "labels":   labels.tolist(),
    }

    print("Gerando HTML …")
    html = build_html(data, unique_labels)

    Path(OUTPUT_HTML).write_text(html, encoding="utf-8")
    print(f"✓ Salvo em '{OUTPUT_HTML}'  ({Path(OUTPUT_HTML).stat().st_size // 1024} KB)")


def build_html(data: dict, unique_labels: list) -> str:
    data_json = json.dumps(data)

    cluster_sections = []
    for lbl in unique_labels:
        label_str = "Noise" if lbl == -1 else f"Cluster {lbl}"
        count = len(data["clusters"][lbl])
        cluster_sections.append(f"""
        <div class="cluster" id="cluster-{lbl}">
            <div class="cluster-header">
                <div class="cluster-title">
                    <span class="cluster-badge">{label_str}</span>
                    <span class="cluster-count">{count} imagens</span>
                </div>
                <div class="cluster-actions">
                    <button class="btn-more" onclick="showMore({lbl})">Ver mais 8 ↓</button>
                    <button class="btn-save" onclick="saveCluster({lbl})">Salvar PNG ↗</button>
                </div>
            </div>
            <div class="image-row" id="row-{lbl}"></div>
        </div>""")

    sections_html = "\n".join(cluster_sections)

    return f"""<!DOCTYPE html>
<html lang="pt">
<head>
<meta charset="UTF-8">
<title>Cluster Viewer — Book Printer Portraits</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: #111;
    color: #e0e0e0;
    font-family: 'Georgia', serif;
    padding: 32px 24px;
  }}

  h1 {{
    font-size: 1.5rem;
    font-weight: normal;
    letter-spacing: 0.08em;
    color: #ccc;
    border-bottom: 1px solid #333;
    padding-bottom: 16px;
    margin-bottom: 32px;
  }}

  .cluster {{
    margin-bottom: 40px;
  }}

  .cluster-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
    padding: 0 4px;
  }}

  .cluster-title {{
    display: flex;
    align-items: baseline;
    gap: 12px;
  }}

  .cluster-badge {{
    font-size: 0.85rem;
    font-weight: bold;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #aaa;
  }}

  .cluster-count {{
    font-size: 0.75rem;
    color: #555;
  }}

  .cluster-actions {{
    display: flex;
    gap: 8px;
  }}

  button {{
    background: #1e1e1e;
    border: 1px solid #333;
    color: #aaa;
    padding: 5px 14px;
    font-size: 0.75rem;
    cursor: pointer;
    border-radius: 3px;
    letter-spacing: 0.05em;
    transition: background 0.15s, color 0.15s;
  }}

  button:hover {{
    background: #2a2a2a;
    color: #ddd;
    border-color: #555;
  }}

  .image-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }}

  .img-card {{
    position: relative;
    cursor: pointer;
    flex-shrink: 0;
  }}

  .img-card img {{
    display: block;
    height: 160px;
    width: auto;
    border: 1px solid #2a2a2a;
    transition: border-color 0.15s;
  }}

  .img-card:hover img {{
    border-color: #666;
  }}

  .img-label {{
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: rgba(0,0,0,0.7);
    color: #888;
    font-size: 9px;
    padding: 2px 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    opacity: 0;
    transition: opacity 0.15s;
  }}

  .img-card:hover .img-label {{
    opacity: 1;
  }}

  /* Lightbox */
  #lightbox {{
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.92);
    z-index: 1000;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 12px;
  }}
  #lightbox.active {{ display: flex; }}
  #lightbox img {{ max-width: 90vw; max-height: 85vh; border: 1px solid #444; }}
  #lightbox-name {{ color: #777; font-size: 0.8rem; }}
  #lightbox-close {{
    position: absolute; top: 16px; right: 24px;
    font-size: 1.8rem; color: #666; cursor: pointer; background: none; border: none;
  }}
</style>
</head>
<body>

<h1>Book Printer Portraits — Cluster Viewer</h1>

{sections_html}

<div id="lightbox">
  <button id="lightbox-close" onclick="closeLightbox()">×</button>
  <img id="lightbox-img" src="" alt="">
  <div id="lightbox-name"></div>
</div>

<script>
const DATA = {data_json};
const PAGE_SIZE = {IMAGES_PER_ROW};
const shown = {{}};  // cluster -> how many shown

function makeCard(imgIdx) {{
  const b64 = DATA.images[imgIdx];
  const name = DATA.names[imgIdx];
  const card = document.createElement('div');
  card.className = 'img-card';
  const img = document.createElement('img');
  img.src = 'data:image/jpeg;base64,' + b64;
  img.title = name;
  img.onclick = () => openLightbox(b64, name);
  const lbl = document.createElement('div');
  lbl.className = 'img-label';
  lbl.textContent = name;
  card.appendChild(img);
  card.appendChild(lbl);
  return card;
}}

function renderCluster(lbl, from, to) {{
  const row = document.getElementById('row-' + lbl);
  const indices = DATA.clusters[lbl];
  const end = Math.min(to, indices.length);
  for (let i = from; i < end; i++) {{
    row.appendChild(makeCard(indices[i]));
  }}
}}

function showMore(lbl) {{
  const from = shown[lbl] || 0;
  const to   = from + PAGE_SIZE;
  renderCluster(lbl, from, to);
  shown[lbl] = to;
  const total = DATA.clusters[lbl].length;
  if (to >= total) {{
    document.querySelector('#cluster-' + lbl + ' .btn-more').disabled = true;
    document.querySelector('#cluster-' + lbl + ' .btn-more').textContent = 'Todas exibidas';
  }}
}}

function saveCluster(lbl) {{
  const row = document.getElementById('row-' + lbl);
  const imgs = row.querySelectorAll('img');
  if (imgs.length === 0) {{ alert('Nenhuma imagem exibida ainda.'); return; }}

  const H = 160;
  const gap = 6;
  const widths = [];
  imgs.forEach(img => {{
    const ratio = H / img.naturalHeight;
    widths.push(Math.round(img.naturalWidth * ratio));
  }});

  const totalW = widths.reduce((a, b) => a + b, 0) + gap * (imgs.length - 1);
  const canvas = document.createElement('canvas');
  canvas.width  = totalW;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, totalW, H);

  let x = 0;
  let loaded = 0;
  imgs.forEach((img, i) => {{
    const tmp = new Image();
    tmp.onload = () => {{
      ctx.drawImage(tmp, x, 0, widths[i], H);
      x += widths[i] + gap;
      loaded++;
      if (loaded === imgs.length) {{
        const a = document.createElement('a');
        a.download = 'cluster_' + lbl + '.png';
        a.href = canvas.toDataURL('image/png');
        a.click();
      }}
    }};
    tmp.src = img.src;
  }});
}}

function openLightbox(b64, name) {{
  document.getElementById('lightbox-img').src = 'data:image/jpeg;base64,' + b64;
  document.getElementById('lightbox-name').textContent = name;
  document.getElementById('lightbox').classList.add('active');
}}

function closeLightbox() {{
  document.getElementById('lightbox').classList.remove('active');
}}

document.getElementById('lightbox').addEventListener('click', function(e) {{
  if (e.target === this) closeLightbox();
}});

// Renderiza primeiros 8 de cada cluster
const labels = {list(unique_labels)};
labels.forEach(lbl => {{
  shown[lbl] = 0;
  showMore(lbl);
}});
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()