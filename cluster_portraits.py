"""
Book Printer Portraits Clustering with OpenCLIP + UMAP
======================================================
Workflow:
  1. Run once → embeddings cached to disk (embeddings_cache.npz)
  2. Tweak CLUSTER CONFIG block → re-run → only clustering + viz re-executed
  3. Outputs: umap_clusters.png  +  cluster_samples/cluster_XX_samples.png
"""

import os
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████████   CLUSTER CONFIG   ███████████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_DIR        = r"D:\HERMES\imagens_crop"         # pasta raiz com subpastas de IDs
METADATA_DIR     = r"D:\HERMES\metadata"         # pasta com JSONs de mesmo nome que o ID
CACHE_FILE       = "embeddings_cache.npz"
OUTPUT_DIR       = "cluster_samples"
UMAP_PLOT_FILE   = "umap_clusters.png"

# --- OpenCLIP model ---
CLIP_MODEL       = "ViT-B-32"          # e.g. "ViT-L-14", "ViT-H-14", "RN50"
CLIP_PRETRAINED  = "laion2b_s34b_b79k" # e.g. "openai", "laion400m_e32"

# --- UMAP ---
UMAP_N_NEIGHBORS = 8                  # larger = more global structure
UMAP_MIN_DIST    = 0.0                 # smaller = tighter clusters in plot
UMAP_METRIC      = "cosine"            # "cosine" | "euclidean" | "correlation"

# --- Clustering algorithm ---
CLUSTER_METHOD   = "hdbscan"           # "hdbscan" | "kmeans" | "agglomerative"

# HDBSCAN (good default — finds number of clusters automatically)
HDBSCAN_MIN_CLUSTER_SIZE  = 10        # increase → fewer, larger clusters
HDBSCAN_MIN_SAMPLES       = 1         # increase → more noise points (-1 label)

# KMeans (fixed number of clusters)
KMEANS_N_CLUSTERS         = 12

# Agglomerative
AGGLO_N_CLUSTERS          = 12
AGGLO_LINKAGE             = "ward"     # "ward" | "complete" | "average"

# --- Visualization ---
THUMB_SIZE       = 48                  # thumbnail size in pixels (UMAP plot)
N_SAMPLES        = 10                  # examples shown per cluster
FIGSIZE_UMAP     = (20, 16)
SEED             = 42

# ─────────────────────────────────────────────────────────────────────────────
# ████████████████████████   PIPELINE   ███████████████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def find_images(directory: str) -> list[Path]:
    """
    Estrutura esperada:
        imagens/
          <ID_1>/   imagem_a.jpg   imagem_b.jpg   (duplicatas — pega só a 1ª)
          <ID_2>/   imagem_a.jpg   imagem_b.jpg
          ...
    Se não houver subpastas, cai no comportamento antigo (varre tudo).
    """
    root = Path(directory)
    paths = []

    subdirs = sorted([d for d in root.iterdir() if d.is_dir()])

    if subdirs:
        for subdir in subdirs:
            imgs = sorted([
                p for p in subdir.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
            ])
            if imgs:
                paths.append(imgs[0])   # primeira imagem de cada ID
            else:
                print(f"  ⚠  Sem imagem em: {subdir}")
    else:
        # fallback: sem subpastas, pega tudo direto
        paths = sorted([
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        ])

    print(f"  Encontradas {len(paths)} imagens (1 por ID) em '{directory}'")
    return paths


# ── 1. EMBEDDINGS ────────────────────────────────────────────────────────────

# Definida fora da função para compatibilidade com multiprocessing no Windows
class ImgDataset:
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        try:
            from PIL import Image as _Image
            img = _Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img), idx
        except Exception:
            import torch as _torch
            return _torch.zeros(3, 224, 224), idx


def compute_embeddings(image_paths: list[Path]) -> np.ndarray:
    """Encode images with OpenCLIP. Returns float32 array (N, D)."""
    import open_clip
    import torch
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED
    )
    model.eval().to(device)

    dataset  = ImgDataset(image_paths, preprocess)
    loader   = DataLoader(dataset, batch_size=64, num_workers=0, pin_memory=False)

    all_embs = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
        for batch_imgs, _ in loader:
            feats = model.encode_image(batch_imgs.to(device))
            feats = feats / feats.norm(dim=-1, keepdim=True)   # L2-normalise
            all_embs.append(feats.cpu().float().numpy())

    return np.vstack(all_embs)


def load_or_compute_embeddings(image_paths: list[Path]) -> np.ndarray:
    if Path(CACHE_FILE).exists():
        print(f"  Loading cached embeddings from '{CACHE_FILE}' …")
        data = np.load(CACHE_FILE, allow_pickle=True)
        cached_paths = list(data["paths"])
        current_paths = [str(p) for p in image_paths]
        if cached_paths == current_paths:
            print(f"  Cache hit — {len(cached_paths)} embeddings loaded.")
            return data["embeddings"]
        else:
            print("  Image list changed — recomputing embeddings …")
    else:
        print("  No cache found — computing embeddings …")

    embeddings = compute_embeddings(image_paths)
    np.savez(CACHE_FILE,
             embeddings=embeddings,
             paths=[str(p) for p in image_paths])
    print(f"  Embeddings saved to '{CACHE_FILE}'.")
    return embeddings


# ── 2. UMAP ──────────────────────────────────────────────────────────────────

def run_umap(embeddings: np.ndarray) -> np.ndarray:
    from umap import UMAP
    print("  Running UMAP …")
    reducer = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=SEED,
        verbose=True,
    )
    return reducer.fit_transform(embeddings)


# ── 3. CLUSTERING ─────────────────────────────────────────────────────────────

def run_clustering(embeddings: np.ndarray) -> np.ndarray:
    method = CLUSTER_METHOD.lower()
    print(f"  Clustering with {method} …")

    if method == "hdbscan":
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES,
            metric="euclidean",
            cluster_selection_method="leaf",
            cluster_selection_epsilon=0.2,
        )
        labels = clusterer.fit_predict(embeddings)

    elif method == "kmeans":
        from sklearn.cluster import MiniBatchKMeans
        labels = MiniBatchKMeans(
            n_clusters=KMEANS_N_CLUSTERS, random_state=SEED, n_init=10
        ).fit_predict(embeddings)

    elif method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        labels = AgglomerativeClustering(
            n_clusters=AGGLO_N_CLUSTERS, linkage=AGGLO_LINKAGE
        ).fit_predict(embeddings)

    else:
        raise ValueError(f"Unknown CLUSTER_METHOD: '{method}'")

    unique, counts = np.unique(labels, return_counts=True)
    print(f"  Clusters found: {len(unique)}  "
          f"(noise points: {(labels == -1).sum()})")
    for u, c in zip(unique, counts):
        label_str = f"cluster {u:>3}" if u >= 0 else "  noise  "
        print(f"    {label_str}: {c} images")
    return labels


# ── 4. UMAP THUMBNAIL PLOT ────────────────────────────────────────────────────

def _load_thumb(path: Path, size: int) -> np.ndarray:
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((size, size), Image.LANCZOS)
        return np.array(img)
    except Exception:
        return np.zeros((size, size, 3), dtype=np.uint8)


def plot_umap_thumbnails(
    umap_coords: np.ndarray,
    labels: np.ndarray,
    image_paths: list[Path],
):
    print("  Rendering UMAP thumbnail plot …")
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))
    label_to_color = {
        lbl: ("lightgrey" if lbl == -1 else cmap(i))
        for i, lbl in enumerate(unique_labels)
    }

    fig, ax = plt.subplots(figsize=FIGSIZE_UMAP, facecolor="#0e0e0e")
    ax.set_facecolor("#0e0e0e")

    # Scatter (colour only, thumbnails on top)
    colors = [label_to_color[l] for l in labels]
    ax.scatter(
        umap_coords[:, 0], umap_coords[:, 1],
        c=colors, s=4, alpha=0.25, linewidths=0,
    )

    # Thumbnails
    zoom = THUMB_SIZE / 224
    for i, (path, xy) in enumerate(zip(image_paths, umap_coords)):
        thumb = _load_thumb(path, THUMB_SIZE)
        im = OffsetImage(thumb, zoom=zoom)
        ab = AnnotationBbox(
            im, xy,
            frameon=True,
            bboxprops=dict(
                edgecolor=label_to_color[labels[i]],
                linewidth=1.5,
                boxstyle="square,pad=0.05",
            ),
            pad=0,
        )
        ax.add_artist(ab)

    # Legend
    patches = [
        mpatches.Patch(
            color=label_to_color[lbl],
            label=("Noise" if lbl == -1 else f"Cluster {lbl}"),
        )
        for lbl in unique_labels
    ]
    ax.legend(
        handles=patches,
        loc="upper left",
        fontsize=8,
        ncol=max(1, len(unique_labels) // 20),
        framealpha=0.4,
        facecolor="#1a1a1a",
        labelcolor="white",
    )

    ax.set_title(
        f"Book Printer Portraits — UMAP  [{CLUSTER_METHOD.upper()}]",
        color="white", fontsize=16, pad=14,
    )
    ax.tick_params(colors="grey")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.tight_layout()
    plt.savefig(UMAP_PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  UMAP plot saved to '{UMAP_PLOT_FILE}'")


# ── 5. CLUSTER SAMPLE GRIDS ───────────────────────────────────────────────────

def plot_cluster_samples(labels: np.ndarray, image_paths: list[Path]):
    out = Path(OUTPUT_DIR)
    out.mkdir(exist_ok=True)

    unique_labels = sorted(l for l in set(labels) if l >= 0)
    print(f"  Saving {N_SAMPLES}-sample grids for {len(unique_labels)} clusters …")

    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        chosen = (
            idx if len(idx) <= N_SAMPLES
            else np.random.default_rng(SEED).choice(idx, N_SAMPLES, replace=False)
        )
        n = len(chosen)
        cols = min(n, 5)
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols,
                                 figsize=(cols * 2.8, rows * 2.8),
                                 facecolor="#111")
        
        # Garante que funcione mesmo se o cluster tiver 1 imagem só
        if n == 1:
            axes = [axes]
        else:
            axes = np.array(axes).flatten()

        for ax in axes:
            ax.set_facecolor("#111")
            # Apaga linhas e números, mas mantém a área do label (embaixo) visível
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        for ax, i in zip(axes, chosen):
            try:
                img = Image.open(image_paths[i]).convert("RGB")
                ax.imshow(img)
            except Exception:
                ax.imshow(np.zeros((224, 224, 3), dtype=np.uint8))
            
            # --- IMPRIME O NOME DA IMAGEM EMBAIXO ---
            ax.set_xlabel(image_paths[i].name, fontsize=7, color="#aaa", labelpad=4, wrap=True)

        fig.suptitle(
            f"Cluster {lbl}  —  {len(idx)} images total",
            color="white", fontsize=13, y=1.01,
        )
        plt.tight_layout()
        save_path = out / f"cluster_{lbl:02d}_samples.png"
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()

    print(f"  Sample grids saved to '{OUTPUT_DIR}/'")


# ── MAIN ──────────────────────────────────────────────────────────────────────

# ── 6. METADATA ANNOTATION ───────────────────────────────────────────────────

def annotate_metadata(image_paths: list[Path], labels) -> None:
    """
    Adiciona o campo 'cluster' em cada JSON de metadata correspondente.
    O JSON é identificado pelo nome do ID (nome da subpasta da imagem).
    """
    import json
    meta_root = Path(METADATA_DIR)
    if not meta_root.exists():
        print(f"  ⚠  METADATA_DIR não encontrado: '{METADATA_DIR}' — pulando anotação.")
        return

    ok = skipped = missing = 0
    for path, label in zip(image_paths, labels):
        # O ID é o nome da subpasta (ex: imagens/1163085642/page_0001.jpg → 1163085642)
        item_id = path.parent.name if path.parent != Path(IMAGE_DIR) else path.stem

        # Procura o JSON com esse nome (qualquer extensão .json)
        candidates = list(meta_root.glob(f"{item_id}.json"))
        if not candidates:
            missing += 1
            continue

        json_path = candidates[0]
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            data['cluster'] = int(label)   # -1 = noise (HDBSCAN)

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            ok += 1
        except Exception as e:
            print(f"  ERR {json_path.name}: {e}")
            skipped += 1

    print(f"  Metadata anotados: {ok} ok | {missing} sem JSON | {skipped} erros")


def main():
    print("\n=== [1/5] Finding images ===")
    image_paths = find_images(IMAGE_DIR)
    if not image_paths:
        raise FileNotFoundError(f"No images found in '{IMAGE_DIR}'")

    print("\n=== [2/5] Embeddings ===")
    embeddings = load_or_compute_embeddings(image_paths)

    print("\n=== [3/5] UMAP ===")
    umap_coords = run_umap(embeddings)

    print("\n=== [4/5] Clustering ===")
    labels = run_clustering(umap_coords)           # cluster in embedding space

    # Salva labels para o viewer HTML
    np.save("cluster_labels.npy", labels)
    print("  Labels salvos em 'cluster_labels.npy'")

    # Anota os JSONs de metadata com o número do cluster
    print("\n=== [+] Annotating metadata JSONs ===")
    annotate_metadata(image_paths, labels)

    print("\n===[5/5] Visualisations ===")
    
    # Cria uma cópia dos labels SÓ para a visualização
    # Assim os JSONs e o numpy array preservam os clusters 1 e 2 originais!
    viz_labels = np.copy(labels)
    viz_labels[(viz_labels == 1) | (viz_labels == 2)] = -1

    plot_umap_thumbnails(umap_coords, viz_labels, image_paths)
    plot_cluster_samples(viz_labels, image_paths)

    print("\n✓ Done!")
    print(f"  UMAP plot    → {UMAP_PLOT_FILE}")
    print(f"  Sample grids → {OUTPUT_DIR}/")
    print(f"  Labels       → cluster_labels.npy")
    print(f"  Rode 4_viewer.py para gerar o HTML interativo")


if __name__ == "__main__":
    main()