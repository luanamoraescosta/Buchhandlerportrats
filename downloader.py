"""
DNB Projekt29 Downloader
========================
Baixa todas as imagens e metadados do projeto29 da Deutsche Nationalbibliothek:
"Mehr als 2.400 Bildnisse von Buchhändlern, Buchdruckern und Verlegern des 17. bis 20. Jahrhunderts"

Dependências:
    pip install sickle lxml requests

Uso:
    python dnb_projekt29_downloader.py

Estrutura de saída:
    dnb_projekt29/
    ├── images/          → imagens .jpg organizadas por record_id
    ├── metadata/        → um .json por registro com todos os metadados
    └── metadata_all.json → todos os metadados num único arquivo
"""

import os
import json
import time
import re
import requests
from pathlib import Path
from lxml import etree
from sickle import Sickle

# ── Configurações ─────────────────────────────────────────────────────────────

OUTPUT_DIR   = Path(r"D:/HERMES/")
IMAGES_DIR   = OUTPUT_DIR / "images"
METADATA_DIR = OUTPUT_DIR / "metadata"

OAI_BASE_URL = "https://services.dnb.de/oai2/repository"
PROJECT_SET  = "dnb:digitalisate-oa:projekt29"

# Redução de qualidade da imagem: 0 = máxima qualidade, 21.6 = menor (como no tutorial)
# Use 0 ou remova o parâmetro para qualidade máxima
IMAGE_QUALITY = 0   # altere para 21.6 se quiser arquivos menores

# Intervalo entre downloads (segundos) — seja gentil com o servidor
SLEEP_BETWEEN_RECORDS = 0.5
SLEEP_BETWEEN_IMAGES  = 0.2

# Retomar de onde parou (True = pula records já baixados)
RESUME = True

# ── Namespaces XML ─────────────────────────────────────────────────────────────

NS = {
    'oai':    'http://www.openarchives.org/OAI/2.0/',
    'oai_dc': 'http://www.openarchives.org/OAI/2.0/oai_dc/',
    'dc':     'http://purl.org/dc/elements/1.1/',
    'mets':   'http://www.loc.gov/METS/',
    'mods':   'http://www.loc.gov/mods/v3',
    'xlink':  'http://www.w3.org/1999/xlink',
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_filename(text: str, max_len: int = 80) -> str:
    """Converte texto para nome de arquivo seguro."""
    text = re.sub(r'[\\/*?:"<>|]', '_', text)
    text = text.strip().replace(' ', '_')
    return text[:max_len]


def xpath_all(tree, path):
    """Retorna lista de textos para um xpath, nunca None."""
    return tree.xpath(path, namespaces=NS) or []


def xpath_first(tree, path, default=""):
    results = xpath_all(tree, path)
    return results[0].strip() if results else default


def extract_dc_metadata(tree):
    """Extrai todos os campos Dublin Core de um record."""
    base = '/oai:record/oai:metadata/oai_dc:dc/'
    fields = [
        'dc:title', 'dc:creator', 'dc:subject', 'dc:description',
        'dc:publisher', 'dc:contributor', 'dc:date', 'dc:type',
        'dc:format', 'dc:identifier', 'dc:source', 'dc:language',
        'dc:relation', 'dc:coverage', 'dc:rights',
    ]
    meta = {}
    for field in fields:
        tag = field.split(':')[1]
        values = xpath_all(tree, f'{base}{field}/text()')
        if values:
            meta[tag] = values if len(values) > 1 else values[0]
    return meta


def extract_oai_header(tree):
    """Extrai o cabeçalho OAI (identifier, datestamp, setSpec)."""
    return {
        'identifier': xpath_first(tree, '/oai:record/oai:header/oai:identifier/text()'),
        'datestamp':  xpath_first(tree, '/oai:record/oai:header/oai:datestamp/text()'),
        'setSpecs':   xpath_all(tree,   '/oai:record/oai:header/oai:setSpec/text()'),
    }


def build_image_url(base_url: str) -> str:
    """Ajusta URL de imagem conforme qualidade configurada."""
    if IMAGE_QUALITY == 0:
        # Remove o parâmetro reduce para qualidade máxima
        return re.sub(r'\?reduce=[\d.]+', '', base_url)
    else:
        return re.sub(r'reduce=[\d.]+', f'reduce={IMAGE_QUALITY}', base_url)


def download_image(url: str, dest: Path, session: requests.Session) -> bool:
    """Baixa uma imagem e salva em dest. Retorna True se bem-sucedido."""
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"    ✗ Erro ao baixar {url}: {e}")
        return False


def get_image_urls_from_record(record_id: str) -> list[str]:
    """
    Consulta a OAI2 pelo record individual (GetRecord) em formato METS/MODS
    para extrair as URLs das imagens digitalizadas.
    Fallback: tenta construir URLs via bookviewer a partir do ID numérico.
    """
    urls = []
    # Tenta GetRecord em METS para obter file locations
    try:
        sickle_single = Sickle(OAI_BASE_URL)
        record = sickle_single.GetRecord(
            identifier=record_id, metadataPrefix='mets'
        )
        tree = etree.ElementTree(record.xml)
        # METS flocat: xlink:href com URLs das imagens
        locs = tree.xpath(
            '//mets:FLocat/@xlink:href', namespaces=NS
        )
        urls = [build_image_url(u) for u in locs if '/img/page/' in u or '.jpg' in u]
    except Exception:
        pass

    # Fallback: bookviewer via DC identifier
    if not urls:
        # O identifier numérico da DNB está no record_id ou nos identifiers DC
        match = re.search(r'(\d{7,})', record_id)
        if match:
            dnb_id = match.group(1)
            # Tenta a URL paginada padrão (até 500 páginas)
            # O bookviewer valida automaticamente e retorna 404 fora do range
            urls = _probe_bookviewer_urls(dnb_id)

    return urls


def _probe_bookviewer_urls(dnb_id: str, max_pages: int = 500) -> list[str]:
    """
    Constrói URLs do bookviewer testando existência das páginas.
    Para projeto29, cada "livro" costuma ser 1-3 imagens (retratos).
    Retorna lista de URLs válidas.
    """
    session = requests.Session()
    urls = []
    for page in range(1, max_pages + 1):
        quality_param = "" if IMAGE_QUALITY == 0 else f"?reduce={IMAGE_QUALITY}"
        url = f"https://portal.dnb.de/bookviewer/view/{dnb_id}/img/page/{page}/p.jpg{quality_param}"
        try:
            r = session.head(url, timeout=10)
            if r.status_code == 200:
                urls.append(url)
            elif r.status_code == 404:
                break  # Fim das páginas
        except Exception:
            break
    return urls


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Cria diretórios
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== DNB Projekt29 Downloader ===")
    print(f"Saída: {OUTPUT_DIR.resolve()}\n")

    sickle = Sickle(OAI_BASE_URL)
    session = requests.Session()
    session.headers.update({'User-Agent': 'DNBLab-Downloader/1.0 (research use)'})

    all_metadata = []
    record_count = 0
    image_count  = 0

    print("Conectando à OAI e listando records do projekt29...")
    records_iter = sickle.ListRecords(
        **{'metadataPrefix': 'oai_dc', 'set': PROJECT_SET}
    )

    for record in records_iter:
        record_count += 1
        tree = etree.ElementTree(record.xml)

        # ── Metadados ──────────────────────────────────────────────────────
        header = extract_oai_header(tree)
        dc_meta = extract_dc_metadata(tree)

        record_id = header['identifier']
        # ID curto para nomes de arquivo
        short_id = record_id.split('/')[-1] if '/' in record_id else record_id
        short_id = safe_filename(short_id)

        title = dc_meta.get('title', short_id)
        if isinstance(title, list):
            title = title[0]

        print(f"\n[{record_count}] {title}")
        print(f"    ID: {record_id}")

        # ── Pasta de imagens do record ──────────────────────────────────────
        record_img_dir = IMAGES_DIR / short_id
        meta_file = METADATA_DIR / f"{short_id}.json"

        # Resume: pula se já existe JSON de metadados
        if RESUME and meta_file.exists():
            print("    → já baixado, pulando.")
            try:
                with open(meta_file) as f:
                    all_metadata.append(json.load(f))
            except Exception:
                pass
            continue

        # ── URLs das imagens ────────────────────────────────────────────────
        # Tenta extrair do METS; usa DC identifier como fallback
        identifiers = dc_meta.get('identifier', [])
        if isinstance(identifiers, str):
            identifiers = [identifiers]

        image_urls = []

        # Método 1: GetRecord METS
        try:
            mets_record = sickle.GetRecord(
                identifier=record_id, metadataPrefix='mets'
            )
            mets_tree = etree.ElementTree(mets_record.xml)
            locs = mets_tree.xpath('//mets:FLocat/@xlink:href', namespaces=NS)
            image_urls = [
                build_image_url(u) for u in locs
                if '/img/page/' in u or u.lower().endswith('.jpg')
            ]
            if image_urls:
                print(f"    → {len(image_urls)} imagem(ns) via METS")
        except Exception as e:
            print(f"    ⚠ METS não disponível: {e}")

        # Método 2: bookviewer probe via identifier numérico
        if not image_urls:
            for ident in identifiers:
                match = re.search(r'(\d{7,})', ident)
                if match:
                    dnb_id = match.group(1)
                    print(f"    → tentando bookviewer probe para ID {dnb_id}...")
                    image_urls = _probe_bookviewer_urls(dnb_id)
                    if image_urls:
                        print(f"    → {len(image_urls)} imagem(ns) via bookviewer")
                        break

        # ── Download das imagens ────────────────────────────────────────────
        downloaded_paths = []
        if image_urls:
            record_img_dir.mkdir(exist_ok=True)
            for i, url in enumerate(image_urls, 1):
                ext = '.jpg'
                img_filename = f"page_{i:04d}{ext}"
                dest = record_img_dir / img_filename
                if RESUME and dest.exists():
                    downloaded_paths.append(str(dest.relative_to(OUTPUT_DIR)))
                    continue
                ok = download_image(url, dest, session)
                if ok:
                    image_count += 1
                    downloaded_paths.append(str(dest.relative_to(OUTPUT_DIR)))
                    print(f"    ✓ {img_filename}")
                time.sleep(SLEEP_BETWEEN_IMAGES)
        else:
            print("    ✗ Nenhuma imagem encontrada para este record.")

        # ── Salva metadados individuais ─────────────────────────────────────
        record_meta = {
            'oai_header': header,
            'dublin_core': dc_meta,
            'image_urls': image_urls,
            'downloaded_images': downloaded_paths,
        }

        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(record_meta, f, ensure_ascii=False, indent=2)

        all_metadata.append(record_meta)

        print(f"    ✓ Metadados salvos em {meta_file}")
        time.sleep(SLEEP_BETWEEN_RECORDS)

    # ── Salva JSON consolidado ──────────────────────────────────────────────
    all_meta_file = OUTPUT_DIR / "metadata_all.json"
    with open(all_meta_file, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"✅ Concluído!")
    print(f"   Records processados : {record_count}")
    print(f"   Imagens baixadas    : {image_count}")
    print(f"   JSON individual     : {METADATA_DIR}/")
    print(f"   JSON consolidado    : {all_meta_file}")


if __name__ == '__main__':
    main()