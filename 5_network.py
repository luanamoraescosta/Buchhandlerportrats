"""
Script 5: Grafo de rede social dos creators (gravadores)
=========================================================
Lê os JSONs de metadata (com campo 'cluster' já preenchido pelo
cluster_portraits.py) e constrói dois grafos interativos em HTML:

  Modo A — por CLUSTER: nós são imagens, agrupados por cluster,
           coloridos por creator. Revela se creators se concentram
           em clusters visuais específicos.

  Modo B — por CREATOR: nós são imagens, agrupados por creator,
           coloridos por cluster. Revela a distribuição visual
           dos trabalhos de cada gravador.

Cada nó exibe a thumbnail da imagem correspondente.
Edges ligam imagens do mesmo creator (modo A) ou do mesmo cluster (modo B).

Uso:
  pip install pyvis networkx
  python 5_network.py
"""

import json
import base64
import re
from pathlib import Path
from io import BytesIO
from PIL import Image
import networkx as nx

# ── CONFIG ────────────────────────────────────────────────────────────────────

METADATA_DIR  = r"D:\HERMES\metadata"
IMAGE_DIR     = r"D:\HERMES\imagens_crop"    # imagens já recortadas
OUTPUT_A      = "network_by_cluster.html"
OUTPUT_B      = "network_by_creator.html"

THUMB_SIZE    = 80     # px — tamanho dos thumbnails nos nós
THUMB_QUALITY = 70
MIN_CONNECTIONS = 2    # ignora creators com menos de N retratos (evita ruído)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# ── CORES ─────────────────────────────────────────────────────────────────────

PALETTE = [
    "#e6194b","#3cb44b","#4363d8","#f58231","#911eb4",
    "#42d4f4","#f032e6","#bfef45","#fabed4","#469990",
    "#dcbeff","#9A6324","#800000","#aaffc3","#808000",
    "#ffd8b1","#000075","#a9a9a9","#ffffff","#000000",
]

def color_for(idx: int) -> str:
    return PALETTE[idx % len(PALETTE)]

# ── HELPERS ───────────────────────────────────────────────────────────────────

def img_to_b64(path: Path) -> str:
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=THUMB_QUALITY)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def find_image(item_id: str) -> Path | None:
    root = Path(IMAGE_DIR)
    for ext in SUPPORTED_EXTS:
        p = root / f"{item_id}{ext}"
        if p.exists():
            return p
    # fallback: subpasta
    subdir = root / item_id
    if subdir.is_dir():
        imgs = sorted([x for x in subdir.iterdir() if x.suffix.lower() in SUPPORTED_EXTS])
        if imgs:
            return imgs[0]
    return None


def parse_creator(creator_raw) -> str:
    """Extrai nome limpo do campo creator (pode ser string ou lista)."""
    if isinstance(creator_raw, list):
        creator_raw = creator_raw[0] if creator_raw else ""
    if not creator_raw:
        return "Unknown"
    # Remove role tags like [Stecher], [Radierer], etc.
    name = re.sub(r'\s*\[.*?\]', '', str(creator_raw)).strip()
    return name if name else "Unknown"


def load_records() -> list[dict]:
    """Lê todos os JSONs de metadata e retorna lista de registros."""
    records = []
    meta_root = Path(METADATA_DIR)
    for json_path in sorted(meta_root.glob("*.json")):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue

        item_id = json_path.stem
        cluster = data.get("cluster", -99)
        if cluster == -99:
            continue  # ainda não foi clusterizado

        dc = data.get("dublin_core", {})
        creator_raw = dc.get("creator", "")
        creator = parse_creator(creator_raw)

        title_raw = dc.get("title", item_id)
        title = title_raw[:60] + "…" if len(title_raw) > 60 else title_raw

        img_path = find_image(item_id)
        b64 = img_to_b64(img_path) if img_path else ""

        records.append({
            "id":      item_id,
            "cluster": int(cluster),
            "creator": creator,
            "title":   title,
            "b64":     b64,
        })

    print(f"  {len(records)} registros carregados")
    return records


# ── HTML BUILDER ──────────────────────────────────────────────────────────────

def build_html(nodes: list[dict], edges: list[tuple],
               title: str, legend_items: list[dict]) -> str:
    """
    Gera HTML standalone com grafo vis.js.
    nodes: [{id, label, color, image_b64, tooltip}]
    edges: [(from_id, to_id)]
    legend_items: [{color, label}]
    """
    nodes_js = json.dumps([{
        "id":    n["id"],
        "label": "",
        "title": n["tooltip"],
        "color": { "background": n["color"], "border": "#333",
                   "highlight": {"background": n["color"], "border": "#fff"} },
        "image": f"data:image/jpeg;base64,{n['image_b64']}" if n["image_b64"] else "",
        "shape": "circularImage" if n["image_b64"] else "dot",
        "size":  32,
        "borderWidth": 2,
    } for n in nodes])

    edges_js = json.dumps([{"from": e[0], "to": e[1], "color": {"color": "#333", "opacity": 0.25}} for e in edges])

    legend_html = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0">'
        f'<div style="width:14px;height:14px;border-radius:50%;background:{item["color"]};flex-shrink:0"></div>'
        f'<span style="font-size:11px;color:#ccc;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{item["label"]}</span>'
        f'</div>'
        for item in legend_items
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/dist/vis-network.min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #111; color: #ddd; font-family: Arial, sans-serif; height: 100vh; display: flex; flex-direction: column; }}
#header {{ padding: 12px 20px; background: #1a1a1a; border-bottom: 1px solid #333; display: flex; align-items: center; justify-content: space-between; }}
#header h1 {{ font-size: 1rem; font-weight: normal; color: #aaa; letter-spacing: 0.05em; }}
#controls {{ display: flex; gap: 8px; }}
button {{ background: #222; border: 1px solid #444; color: #aaa; padding: 5px 12px; font-size: 0.75rem; cursor: pointer; border-radius: 3px; }}
button:hover {{ background: #333; color: #eee; }}
#main {{ display: flex; flex: 1; overflow: hidden; }}
#network {{ flex: 1; }}
#sidebar {{ width: 200px; background: #161616; border-left: 1px solid #222; padding: 12px; overflow-y: auto; }}
#sidebar h2 {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; color: #555; margin-bottom: 10px; }}
#info {{ margin-top: 20px; padding-top: 12px; border-top: 1px solid #222; }}
#info img {{ width: 100%; border-radius: 4px; margin-bottom: 6px; }}
#info p {{ font-size: 10px; color: #888; line-height: 1.4; }}
</style>
</head>
<body>
<div id="header">
  <h1>{title}</h1>
  <div id="controls">
    <button onclick="network.fit()">Fit view</button>
    <button onclick="togglePhysics()">Toggle physics</button>
  </div>
</div>
<div id="main">
  <div id="network"></div>
  <div id="sidebar">
    <h2>Legend</h2>
    <div id="legend">{legend_html}</div>
    <div id="info"><p style="color:#555;font-size:10px">Click a node to see details</p></div>
  </div>
</div>
<script>
const nodes = new vis.DataSet({nodes_js});
const edges = new vis.DataSet({edges_js});
const container = document.getElementById('network');
let physicsOn = true;
const options = {{
  nodes: {{ borderWidth: 2, chosen: {{ node: (v,id,s,h) => {{ v.borderWidth = 4; }} }} }},
  edges: {{ smooth: {{ type: 'continuous' }}, width: 1 }},
  physics: {{
    enabled: true,
    forceAtlas2Based: {{ gravitationalConstant: -80, centralGravity: 0.01, springLength: 120, springConstant: 0.08 }},
    solver: 'forceAtlas2Based',
    stabilization: {{ iterations: 300 }}
  }},
  interaction: {{ hover: true, tooltipDelay: 100 }}
}};
const network = new vis.Network(container, {{ nodes, edges }}, options);
network.on('click', params => {{
  if (!params.nodes.length) return;
  const node = nodes.get(params.nodes[0]);
  const info = document.getElementById('info');
  info.innerHTML = node.image
    ? `<img src="${{node.image}}"><p>${{node.title}}</p>`
    : `<p>${{node.title}}</p>`;
}});
function togglePhysics() {{
  physicsOn = !physicsOn;
  network.setOptions({{ physics: {{ enabled: physicsOn }} }});
}}
</script>
</body>
</html>"""


# ── MODO A: por cluster (edges = mesmo creator) ───────────────────────────────

def build_network_by_cluster(records: list[dict]) -> str:
    # Filtra creators com poucos retratos
    from collections import Counter
    creator_counts = Counter(r["creator"] for r in records)
    valid_creators = {c for c, n in creator_counts.items() if n >= MIN_CONNECTIONS}

    # Cores por creator
    creators_sorted = sorted(valid_creators)
    creator_color = {c: color_for(i) for i, c in enumerate(creators_sorted)}
    creator_color["Unknown"] = "#555"

    nodes = []
    for r in records:
        color = creator_color.get(r["creator"], "#555")
        nodes.append({
            "id":       r["id"],
            "color":    color,
            "image_b64": r["b64"],
            "tooltip":  f"<b>{r['creator']}</b><br>Cluster {r['cluster']}<br><small>{r['title']}</small>",
        })

    # Edges: liga imagens do mesmo creator (dentro de valid_creators)
    from itertools import combinations
    creator_groups = {}
    for r in records:
        if r["creator"] in valid_creators:
            creator_groups.setdefault(r["creator"], []).append(r["id"])

    edges = []
    for ids in creator_groups.values():
        for a, b in combinations(ids, 2):
            edges.append((a, b))

    legend = [{"color": creator_color[c], "label": c} for c in creators_sorted[:30]]

    return build_html(nodes, edges,
                      "Network by Cluster — edges = same creator",
                      legend)


# ── MODO B: por creator (edges = mesmo cluster) ───────────────────────────────

def build_network_by_creator(records: list[dict]) -> str:
    # Cores por cluster
    clusters = sorted(set(r["cluster"] for r in records))
    cluster_color = {c: ("#555" if c == -1 else color_for(i))
                     for i, c in enumerate(clusters)}

    nodes = []
    for r in records:
        color = cluster_color.get(r["cluster"], "#555")
        nodes.append({
            "id":       r["id"],
            "color":    color,
            "image_b64": r["b64"],
            "tooltip":  f"<b>{r['creator']}</b><br>Cluster {r['cluster']}<br><small>{r['title']}</small>",
        })

    # Edges: liga imagens do mesmo cluster
    from itertools import combinations
    cluster_groups = {}
    for r in records:
        cluster_groups.setdefault(r["cluster"], []).append(r["id"])

    edges = []
    for cluster_id, ids in cluster_groups.items():
        if cluster_id == -1:
            continue   # não liga noise
        # Limita edges por cluster pra não poluir (máximo 100 por cluster)
        pairs = list(combinations(ids, 2))
        import random
        random.seed(42)
        if len(pairs) > 100:
            pairs = random.sample(pairs, 100)
        edges.extend(pairs)

    legend = [{"color": cluster_color[c],
               "label": "Noise" if c == -1 else f"Cluster {c}"}
              for c in clusters]

    return build_html(nodes, edges,
                      "Network by Creator — edges = same cluster",
                      legend)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Carregando metadata ===")
    records = load_records()
    if not records:
        print("Nenhum registro com campo 'cluster' encontrado.")
        print("Rode cluster_portraits.py primeiro.")
        return

    print(f"\n=== Gerando {OUTPUT_A} ===")
    html_a = build_network_by_cluster(records)
    Path(OUTPUT_A).write_text(html_a, encoding="utf-8")
    print(f"  Salvo → {OUTPUT_A}")

    print(f"\n=== Gerando {OUTPUT_B} ===")
    html_b = build_network_by_creator(records)
    Path(OUTPUT_B).write_text(html_b, encoding="utf-8")
    print(f"  Salvo → {OUTPUT_B}")

    print("\n✓ Abra os HTMLs no browser. Não precisa de servidor.")


if __name__ == "__main__":
    main()