"""
DNB Metadata Enricher
=====================
Complementa os metadados já baixados pelo dnb_projekt29_downloader.py
consultando formatos mais ricos da OAI:
  - METS/MODS  → campos detalhados (nomes, datas, GND, notas, física do objeto...)
  - RDFxml     → linked data com relações entre entidades

Pré-requisitos:
    pip install sickle lxml requests

Uso:
    python dnb_enrich_metadata.py

O script lê cada arquivo em dnb_projekt29/metadata/*.json,
consulta a OAI pelos formatos adicionais e adiciona os dados novos
sem sobrescrever o que já existe.
"""

import json
import time
import requests
from pathlib import Path
from lxml import etree
from sickle import Sickle

# ── Configurações ──────────────────────────────────────────────────────────────

METADATA_DIR  = Path("D:/HERMES/metadata")
OUTPUT_DIR    = Path("D:/HERMES/dnb_projekt29")
OAI_BASE_URL  = "https://services.dnb.de/oai2/repository"

SLEEP_BETWEEN = 0.5   # segundos entre chamadas à API

# ── Namespaces ─────────────────────────────────────────────────────────────────

NS = {
    'oai':     'http://www.openarchives.org/OAI/2.0/',
    'mets':    'http://www.loc.gov/METS/',
    'mods':    'http://www.loc.gov/mods/v3',
    'xlink':   'http://www.w3.org/1999/xlink',
    'rdf':     'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'dc':      'http://purl.org/dc/elements/1.1/',
    'dcterms': 'http://purl.org/dc/terms/',
    'gndo':    'https://d-nb.info/standards/elementset/gnd#',
    'owl':     'http://www.w3.org/2002/07/owl#',
    'skos':    'http://www.w3.org/2004/02/skos/core#',
    'rdfs':    'http://www.w3.org/2000/01/rdf-schema#',
    'bibo':    'http://purl.org/ontology/bibo/',
    'foaf':    'http://xmlns.com/foaf/0.1/',
    'isbd':    'http://iflastandards.info/ns/isbd/elements/',
    # RDA Unconstrained properties — namespace usado pela DNB no RDFxml
    'rdau':     'http://rdaregistry.info/Elements/u/',
    'rdaw':     'http://rdaregistry.info/Elements/w/',
    'rdae':     'http://rdaregistry.info/Elements/e/',
    'rdam':     'http://rdaregistry.info/Elements/m/',
    'rdai':     'http://rdaregistry.info/Elements/i/',
    # Papéis de agentes MARC (gravador, artista, editor, etc.)
    'marcRole': 'http://id.loc.gov/vocabulary/relators/',
    # Umbel / POWDER — links e descrições
    'umbel':    'http://umbel.org/umbel#',
    'wdrs':     'http://www.w3.org/2007/05/powder-s#',
}

def xp(tree, path, first=False, default=""):
    results = tree.xpath(path, namespaces=NS)
    if not results:
        return default if first else []
    return results[0].strip() if first else [r.strip() if isinstance(r, str) else r for r in results]


# ── Extratores por formato ─────────────────────────────────────────────────────

def extract_mods(tree):
    """
    Extrai campos MODS da resposta GetRecord em formato 'mets'.
    MODS é encapsulado dentro do METS como dmdSec.
    """
    meta = {}

    # ── Títulos ──────────────────────────────────────────────────────────────
    meta['titles'] = {
        'main':      xp(tree, '//mods:titleInfo[not(@type)]/mods:title/text()'),
        'alternative': xp(tree, '//mods:titleInfo[@type="alternative"]/mods:title/text()'),
        'subtitle':  xp(tree, '//mods:titleInfo/mods:subTitle/text()'),
    }

    # ── Pessoas / Nomes ───────────────────────────────────────────────────────
    names = []
    for name_el in tree.xpath('//mods:name', namespaces=NS):
        n = {
            'type':        name_el.get('type', ''),
            'authority':   name_el.get('authority', ''),
            'value_uri':   name_el.get('valueURI', ''),   # link GND
            'display':     xp(name_el, 'mods:displayForm/text()', first=True),
            'family':      xp(name_el, 'mods:namePart[@type="family"]/text()', first=True),
            'given':       xp(name_el, 'mods:namePart[@type="given"]/text()', first=True),
            'date':        xp(name_el, 'mods:namePart[@type="date"]/text()', first=True),
            'roles':       xp(name_el, 'mods:role/mods:roleTerm/text()'),
            'description': xp(name_el, 'mods:description/text()', first=True),
        }
        # Remove campos vazios
        n = {k: v for k, v in n.items() if v}
        if n:
            names.append(n)
    if names:
        meta['names'] = names

    # ── Tipo de recurso ───────────────────────────────────────────────────────
    meta['resource_type'] = xp(tree, '//mods:typeOfResource/text()')

    # ── Gênero / Forma ────────────────────────────────────────────────────────
    meta['genre'] = xp(tree, '//mods:genre/text()')

    # ── Origem / Publicação ───────────────────────────────────────────────────
    origin = {}
    origin['place']      = xp(tree, '//mods:originInfo/mods:place/mods:placeTerm/text()')
    origin['publisher']  = xp(tree, '//mods:originInfo/mods:publisher/text()')
    origin['date_issued']   = xp(tree, '//mods:originInfo/mods:dateIssued/text()')
    origin['date_created']  = xp(tree, '//mods:originInfo/mods:dateCreated/text()')
    origin['date_captured'] = xp(tree, '//mods:originInfo/mods:dateCaptured/text()')
    origin['edition']    = xp(tree, '//mods:originInfo/mods:edition/text()')
    origin['issuance']   = xp(tree, '//mods:originInfo/mods:issuance/text()')
    origin['frequency']  = xp(tree, '//mods:originInfo/mods:frequency/text()')
    meta['origin_info'] = {k: v for k, v in origin.items() if v}

    # ── Idioma ────────────────────────────────────────────────────────────────
    meta['language'] = xp(tree, '//mods:language/mods:languageTerm/text()')

    # ── Descrição física ─────────────────────────────────────────────────────
    phys = {}
    phys['extent']       = xp(tree, '//mods:physicalDescription/mods:extent/text()')
    phys['form']         = xp(tree, '//mods:physicalDescription/mods:form/text()')
    phys['media_type']   = xp(tree, '//mods:physicalDescription/mods:internetMediaType/text()')
    phys['digital_origin'] = xp(tree, '//mods:physicalDescription/mods:digitalOrigin/text()')
    phys['note']         = xp(tree, '//mods:physicalDescription/mods:note/text()')
    meta['physical_description'] = {k: v for k, v in phys.items() if v}

    # ── Resumo / Abstract ─────────────────────────────────────────────────────
    meta['abstract'] = xp(tree, '//mods:abstract/text()')

    # ── Notas ─────────────────────────────────────────────────────────────────
    notes = []
    for note_el in tree.xpath('//mods:note', namespaces=NS):
        note = {'text': (note_el.text or '').strip()}
        if note_el.get('type'):
            note['type'] = note_el.get('type')
        if note['text']:
            notes.append(note)
    if notes:
        meta['notes'] = notes

    # ── Assuntos / Tópicos ────────────────────────────────────────────────────
    subjects = []
    for subj_el in tree.xpath('//mods:subject', namespaces=NS):
        s = {
            'authority':  subj_el.get('authority', ''),
            'value_uri':  subj_el.get('valueURI', ''),
            'topics':     xp(subj_el, 'mods:topic/text()'),
            'geographic': xp(subj_el, 'mods:geographic/text()'),
            'temporal':   xp(subj_el, 'mods:temporal/text()'),
            'name':       xp(subj_el, 'mods:name/mods:displayForm/text()'),
            'genre':      xp(subj_el, 'mods:genre/text()'),
        }
        s = {k: v for k, v in s.items() if v}
        if s:
            subjects.append(s)
    if subjects:
        meta['subjects'] = subjects

    # ── Classificação ─────────────────────────────────────────────────────────
    meta['classification'] = []
    for cls_el in tree.xpath('//mods:classification', namespaces=NS):
        meta['classification'].append({
            'authority': cls_el.get('authority', ''),
            'value':     (cls_el.text or '').strip(),
        })

    # ── Identificadores ───────────────────────────────────────────────────────
    identifiers = {}
    for id_el in tree.xpath('//mods:identifier', namespaces=NS):
        id_type = id_el.get('type', 'unknown')
        val = (id_el.text or '').strip()
        if val:
            if id_type in identifiers:
                if isinstance(identifiers[id_type], list):
                    identifiers[id_type].append(val)
                else:
                    identifiers[id_type] = [identifiers[id_type], val]
            else:
                identifiers[id_type] = val
    meta['identifiers'] = identifiers

    # ── Localização / URL de acesso ───────────────────────────────────────────
    meta['urls'] = xp(tree, '//mods:location/mods:url/text()')
    meta['shelf_locator'] = xp(tree, '//mods:location/mods:shelfLocator/text()')

    # ── Partes relacionadas ───────────────────────────────────────────────────
    related = []
    for rel_el in tree.xpath('//mods:relatedItem', namespaces=NS):
        r = {
            'type':  rel_el.get('type', ''),
            'title': xp(rel_el, 'mods:titleInfo/mods:title/text()', first=True),
            'identifier': xp(rel_el, 'mods:identifier/text()', first=True),
        }
        r = {k: v for k, v in r.items() if v}
        if r:
            related.append(r)
    if related:
        meta['related_items'] = related

    # ── Direitos de acesso ────────────────────────────────────────────────────
    meta['access_condition'] = xp(tree, '//mods:accessCondition/text()')

    # ── Record Info ───────────────────────────────────────────────────────────
    meta['record_info'] = {
        'creation_date':     xp(tree, '//mods:recordInfo/mods:recordCreationDate/text()', first=True),
        'change_date':       xp(tree, '//mods:recordInfo/mods:recordChangeDate/text()', first=True),
        'identifier':        xp(tree, '//mods:recordInfo/mods:recordIdentifier/text()', first=True),
        'content_source':    xp(tree, '//mods:recordInfo/mods:recordContentSource/text()', first=True),
        'origin_description': xp(tree, '//mods:recordInfo/mods:recordOrigin/text()', first=True),
    }

    # ── METS: estrutura lógica (ordem das páginas) ────────────────────────────
    struct_map = []
    for div in tree.xpath('//mets:structMap[@TYPE="LOGICAL"]//mets:div', namespaces=NS):
        struct_map.append({
            'type':  div.get('TYPE', ''),
            'label': div.get('LABEL', ''),
            'order': div.get('ORDER', ''),
        })
    if struct_map:
        meta['logical_structure'] = struct_map

    # ── METS: arquivos (imagens) ──────────────────────────────────────────────
    file_list = []
    for file_el in tree.xpath('//mets:fileGrp/mets:file', namespaces=NS):
        flocat = file_el.find('mets:FLocat', NS)
        if flocat is not None:
            file_list.append({
                'id':       file_el.get('ID', ''),
                'mimetype': file_el.get('MIMETYPE', ''),
                'use':      file_el.getparent().get('USE', ''),
                'href':     flocat.get('{http://www.w3.org/1999/xlink}href', ''),
            })
    if file_list:
        meta['mets_files'] = file_list

    # Remove listas vazias do resultado final
    return {k: v for k, v in meta.items() if v not in ([], {}, '')}


def extract_rdf(tree):
    """
    Extrai dados de linked data (RDFxml) da DNB.
    Baseado no Turtle real dos registros do projekt29.
    Cobre: dc/dcterms, rdau (RDA Unconstrained), marcRole,
           umbel, wdrs, GND (gndo), SKOS, RDFS, FOAF.
    """
    meta = {}

    # ── Identificação ─────────────────────────────────────────────────────────
    meta['dnb_uri']        = xp(tree, '//@rdf:about[contains(., "d-nb.info")]')
    meta['same_as']        = xp(tree, '//owl:sameAs/@rdf:resource')
    meta['described_by']   = xp(tree, '//wdrs:describedby/@rdf:resource')
    meta['is_like']        = xp(tree, '//umbel:isLike/@rdf:resource')  # URN de acesso
    meta['dc_identifier']  = xp(tree, '//dc:identifier/text()')

    # ── Título ────────────────────────────────────────────────────────────────
    meta['dc_title']       = xp(tree, '//dc:title/text()')
    meta['rda_title']      = xp(tree, '//rdau:P60365/text()')  # title proper

    # ── Datas ─────────────────────────────────────────────────────────────────
    # P60527 = date of resource (Entstehungszeit — ex: "1800")
    meta['rda_date_of_resource']    = xp(tree, '//rdau:P60527/text()')
    # dcterms:issued = data de publicação/digitalização (ex: "2018")
    meta['dcterms_issued']          = xp(tree, '//dcterms:issued/text()')
    meta['dcterms_modified']        = xp(tree, '//dcterms:modified/text()')
    meta['dc_date']                 = xp(tree, '//dc:date/text()')
    meta['rda_date_of_capture']     = xp(tree, '//rdau:P60074/text()')
    meta['rda_date_of_production']  = xp(tree, '//rdau:P60071/text()')
    meta['rda_date_of_publication'] = xp(tree, '//rdau:P60073/text()')

    # ── Técnica / Material / Forma ────────────────────────────────────────────
    # P60493 = production method / technique (ex: "Schabkunst", "Lithografie")
    meta['rda_technique']           = xp(tree, '//rdau:P60493/text()')
    meta['rda_technique_uri']       = xp(tree, '//rdau:P60493/@rdf:resource')
    # P60159 = applied material
    meta['rda_applied_material']    = xp(tree, '//rdau:P60159/text()')
    meta['rda_applied_material_uri']= xp(tree, '//rdau:P60159/@rdf:resource')
    # P60558 = colour content
    meta['rda_colour_content']      = xp(tree, '//rdau:P60558/@rdf:resource')
    # P60596 = base material
    meta['rda_base_material']       = xp(tree, '//rdau:P60596/@rdf:resource')

    # ── Tipo de objeto ────────────────────────────────────────────────────────
    # P60049 aparece duas vezes: URI de tipo RDA + URI GND de gênero
    meta['rda_content_type']        = xp(tree, '//rdau:P60049/@rdf:resource')
    meta['rda_content_type_text']   = xp(tree, '//rdau:P60049/text()')
    meta['rda_carrier_type']        = xp(tree, '//rdau:P60048/@rdf:resource')
    meta['rda_media_type']          = xp(tree, '//rdau:P60050/@rdf:resource')
    meta['dcterms_medium']          = xp(tree, '//dcterms:medium/@rdf:resource')

    # ── Dimensões / Extensão ──────────────────────────────────────────────────
    meta['rda_extent']              = xp(tree, '//rdau:P60550/text()')
    meta['rda_dimensions']          = xp(tree, '//rdau:P60539/text()')
    meta['dcterms_extent']          = xp(tree, '//dcterms:extent/text()')

    # ── Notas / Inscrições na obra ────────────────────────────────────────────
    # P60327 = note on statement of responsibility (inscrições, assinaturas)
    meta['rda_note_responsibility'] = xp(tree, '//rdau:P60327/text()')
    # P60333 = production statement (lugar : editor formatado)
    meta['rda_production_statement']= xp(tree, '//rdau:P60333/text()')
    meta['rda_edition']             = xp(tree, '//rdau:P60329/text()')

    # ── Publicação / Lugar ────────────────────────────────────────────────────
    meta['rda_place_of_publication']= xp(tree, '//rdau:P60163/text()')
    meta['rda_publisher_name']      = xp(tree, '//rdau:P60547/text()')
    meta['dc_publisher']            = xp(tree, '//dc:publisher/text()')

    # ── Relações / Coleções ───────────────────────────────────────────────────
    # P60469 = is part of manifestation (coleção pai — ex: Grafische Sammlung)
    meta['rda_part_of']             = xp(tree, '//rdau:P60469/@rdf:resource')
    meta['rda_contained_in']        = xp(tree, '//rdau:P60101/@rdf:resource')
    meta['rda_contains']            = xp(tree, '//rdau:P60249/@rdf:resource')
    meta['rda_reproduction_of']     = xp(tree, '//rdau:P60297/@rdf:resource')
    meta['rda_reproduction_of_text']= xp(tree, '//rdau:P60297/text()')
    meta['dcterms_isPartOf']        = xp(tree, '//dcterms:isPartOf/@rdf:resource')
    meta['dcterms_hasPart']         = xp(tree, '//dcterms:hasPart/@rdf:resource')

    # ── Agentes / Papéis MARC (marcRole:) ────────────────────────────────────
    # Os papéis vêm como URIs GND — ctb=contribuidor, art=artista,
    # pbl=publisher, egr=gravador (engraver), prt=impressor, etc.
    meta['role_contributor']        = xp(tree, '//marcRole:ctb/@rdf:resource')
    meta['role_artist']             = xp(tree, '//marcRole:art/@rdf:resource')
    meta['role_publisher']          = xp(tree, '//marcRole:pbl/@rdf:resource')
    meta['role_engraver']           = xp(tree, '//marcRole:egr/@rdf:resource')
    meta['role_printer']            = xp(tree, '//marcRole:prt/@rdf:resource')
    meta['role_creator']            = xp(tree, '//marcRole:cre/@rdf:resource')
    meta['role_lithographer']       = xp(tree, '//marcRole:lth/@rdf:resource')
    meta['role_illustrator']        = xp(tree, '//marcRole:ill/@rdf:resource')
    meta['role_photographer']       = xp(tree, '//marcRole:pht/@rdf:resource')
    meta['role_subject']            = xp(tree, '//marcRole:dpt/@rdf:resource')  # depicted person
    # Captura genérica: qualquer papel marcRole não listado acima
    all_marc = tree.xpath('//marcRole:*/@rdf:resource', namespaces=NS)
    known = set(sum([v for v in [
        meta.get('role_contributor'), meta.get('role_artist'),
        meta.get('role_publisher'),   meta.get('role_engraver'),
        meta.get('role_printer'),     meta.get('role_creator'),
        meta.get('role_lithographer'),meta.get('role_illustrator'),
        meta.get('role_photographer'),meta.get('role_subject'),
    ] if v], []))
    extra = [u for u in all_marc if u not in known]
    if extra:
        meta['role_other'] = extra

    # ── Direitos / Licença ────────────────────────────────────────────────────
    meta['dcterms_license']         = xp(tree, '//dcterms:license/@rdf:resource')
    meta['dc_rights']               = xp(tree, '//dc:rights/text()')

    # ── Idioma ────────────────────────────────────────────────────────────────
    meta['dcterms_language']        = xp(tree, '//dcterms:language/@rdf:resource')
    meta['dc_language']             = xp(tree, '//dc:language/text()')

    # ── GND — pessoa retratada ────────────────────────────────────────────────
    meta['gnd_preferred_name']      = xp(tree, '//gndo:preferredNameForThePerson/text()')
    meta['gnd_variant_names']       = xp(tree, '//gndo:variantNameForThePerson/text()')
    meta['gnd_birth_date']          = xp(tree, '//gndo:dateOfBirth/text()')
    meta['gnd_death_date']          = xp(tree, '//gndo:dateOfDeath/text()')
    meta['gnd_birth_place']         = xp(tree, '//gndo:placeOfBirth/@rdf:resource')
    meta['gnd_death_place']         = xp(tree, '//gndo:placeOfDeath/@rdf:resource')
    meta['gnd_profession']          = xp(tree, '//gndo:professionOrOccupation/@rdf:resource')
    meta['gnd_gender']              = xp(tree, '//gndo:gender/@rdf:resource')
    meta['gnd_related_person']      = xp(tree, '//gndo:relatedPerson/@rdf:resource')
    meta['gnd_broader_term']        = xp(tree, '//gndo:broaderTermInstantial/@rdf:resource')
    meta['gnd_work_name']           = xp(tree, '//gndo:preferredNameForTheWork/text()')

    # ── SKOS ──────────────────────────────────────────────────────────────────
    meta['skos_pref_label']         = xp(tree, '//skos:prefLabel/text()')
    meta['skos_alt_label']          = xp(tree, '//skos:altLabel/text()')
    meta['skos_note']               = xp(tree, '//skos:note/text()')
    meta['skos_broader']            = xp(tree, '//skos:broader/@rdf:resource')

    # ── RDFS / FOAF ───────────────────────────────────────────────────────────
    meta['rdfs_label']              = xp(tree, '//rdfs:label/text()')
    meta['rdfs_comment']            = xp(tree, '//rdfs:comment/text()')
    meta['foaf_depiction']          = xp(tree, '//foaf:depiction/@rdf:resource')
    meta['foaf_thumbnail']          = xp(tree, '//foaf:thumbnail/@rdf:resource')
    meta['foaf_name']               = xp(tree, '//foaf:name/text()')

    # Remove campos vazios
    return {k: v for k, v in meta.items() if v not in ([], {}, '')}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    json_files = sorted(METADATA_DIR.glob("*.json"))
    total = len(json_files)

    if total == 0:
        print(f"Nenhum arquivo JSON encontrado em {METADATA_DIR}")
        return

    print(f"=== DNB Metadata Enricher ===")
    print(f"Enriquecendo {total} registros...\n")

    sickle = Sickle(OAI_BASE_URL)
    session = requests.Session()
    session.headers.update({'User-Agent': 'DNBLab-Enricher/1.0 (research use)'})
    enriched_count = 0
    error_count    = 0

    for i, json_path in enumerate(json_files, 1):
        with open(json_path, encoding='utf-8') as f:
            record = json.load(f)

        record_id = record.get('oai_header', {}).get('identifier', '')
        title = ""
        dc = record.get('dublin_core', {})
        if 'title' in dc:
            title = dc['title'] if isinstance(dc['title'], str) else dc['title'][0]

        print(f"[{i}/{total}] {title or record_id}")

        # Pula se já foi completamente enriquecido com a versão nova (inclui rdau)
        rdf_completo = 'rdf' in record and 'rda_date_of_resource' in record.get('rdf', {})
        if 'mods' in record and rdf_completo:
            print("  → já enriquecido, pulando.")
            continue
        # Se tem rdf antigo (sem campos rdau), apaga para re-enriquecer
        if 'rdf' in record and not rdf_completo:
            print("  → rdf incompleto (versão antiga sem rdau), re-enriquecendo...")
            del record['rdf']

        if not record_id:
            print("  ✗ sem identifier, pulando.")
            continue

        # ── METS/MODS ──────────────────────────────────────────────────────
        if 'mods' not in record:
            try:
                mets_record = sickle.GetRecord(
                    identifier=record_id, metadataPrefix='mets'
                )
                mets_tree = etree.ElementTree(mets_record.xml)
                record['mods'] = extract_mods(mets_tree)
                print(f"  ✓ MODS: {len(record['mods'])} campos")
            except Exception as e:
                print(f"  ⚠ METS/MODS: {e}")
            time.sleep(SLEEP_BETWEEN)

        # ── RDF — fetch direto em d-nb.info/{ID}/about/rdf ───────────────
        # O formato RDFxml NÃO está disponível via OAI para esses registros.
        # O endpoint correto é a URL linked data da DNB diretamente.
        if 'rdf' not in record:
            # Extrai apenas o ID numérico final do identifier OAI
            # Ex: "oai:dnb.de:projekt29/1163071994" → "1163071994"
            # Ex: "oai:dnb.de:1163071994"           → "1163071994"
            import re as _re
            _match = _re.search(r'(\d{7,})', record_id)
            dnb_num_id = _match.group(1) if _match else record_id.split(':')[-1].strip()
            rdf_url = f"https://d-nb.info/{dnb_num_id}/about/rdf"
            try:
                resp = session.get(rdf_url, timeout=20,
                                   headers={'Accept': 'application/rdf+xml'})
                resp.raise_for_status()
                rdf_tree = etree.ElementTree(etree.fromstring(resp.content))
                record['rdf'] = extract_rdf(rdf_tree)
                print(f"  ✓ RDF: {len(record['rdf'])} campos")
            except Exception as e:
                print(f"  ⚠ RDF ({rdf_url}): {e}")
            time.sleep(SLEEP_BETWEEN)

        # ── Salva de volta ─────────────────────────────────────────────────
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        enriched_count += 1

    # ── Regenera o JSON consolidado ────────────────────────────────────────
    print("\nRegenerando metadata_all.json...")
    all_metadata = []
    for json_path in sorted(METADATA_DIR.glob("*.json")):
        try:
            with open(json_path, encoding='utf-8') as f:
                all_metadata.append(json.load(f))
        except Exception:
            pass

    all_meta_file = OUTPUT_DIR / "metadata_all.json"
    with open(all_meta_file, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"✅ Concluído!")
    print(f"   Registros enriquecidos : {enriched_count}")
    print(f"   Erros                  : {error_count}")
    print(f"   JSON consolidado       : {all_meta_file}")


if __name__ == '__main__':
    main()