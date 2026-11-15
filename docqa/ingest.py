"""PDF ingestion: pick an extraction method per content type, page by page.

  prose      -> PDF text layer (PyMuPDF), heading-aware chunks that never cross a page
  tables     -> PyMuPDF table finder (it also handles shaded-row tables without ruling lines);
                one Markdown chunk per table with its caption, continuation pages linked
  charts, infographics, diagrams (embedded images, or large vector drawings)
             -> the region is cropped at high resolution and read by a vision model
  scanned pages (one big image, no text layer)
             -> the whole page is transcribed by the vision model
Running headers/footers are removed. Every chunk keeps its page number for citation.
"""

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from . import llm
from .chunk import Chunk
from .config import settings
from .store import ChunkStore

MIN_IMAGE_AREA = 0.02      # ignore icons and logos smaller than 2% of the page
SCAN_IMAGE_AREA = 0.5      # a page that is mostly one image...
SCAN_MAX_TEXT = 50         # ...with almost no text layer is treated as a scan
VECTOR_FIGURE_AREA = 0.08  # large clusters of vector drawings that are not tables are treated as figures
FIGURE_DPI = 200
VISION_PROMPT_VERSION = "v1"  # part of the cache key: bump when the prompts change

_CAPTION = re.compile(r"^(table|figure|fig\.|chart|exhibit)\s*\d+", re.I)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9•])")

FIGURE_SYSTEM = """You turn one figure from a company report into searchable text. Analysts will ask questions
whose answers may exist only in this figure, so be exhaustive and literal.
- Copy every number, unit, label, legend entry, category, year and caption exactly as printed.
- Charts: give the chart type, title, axes and units, then every data point as "label: value". Only report
  printed values; if a value must be read off an axis, mark it "approx.".
- Infographics and KPI tiles: one line per tile, "metric: value", including any comparison shown.
- Diagrams and maps: list every node, then every connection as "A - B: label" (e.g. "A - B: 27 trucks/day"),
  and what colours, highlighted nodes or dashed lines mean if the figure shows it.
- Purely decorative images (patterns, photos without facts, logos): set informative to false and copy any
  readable text, such as a tagline, into content.
Do not add interpretation, trends or numbers that are not visible. The surrounding page text is context only."""

FIGURE_SCHEMA = {
    "type": "object",
    "properties": {
        "informative": {"type": "boolean"},
        "kind": {"type": "string", "enum": ["chart", "infographic", "diagram", "map", "table", "photo", "other"]},
        "title": {"type": "string", "description": "Printed caption or title; a short description if none."},
        "content": {"type": "string"},
    },
    "required": ["informative", "kind", "title", "content"],
    "additionalProperties": False,
}

SCAN_SYSTEM = """You transcribe a scanned page from a company report so it can be searched and quoted.
Transcribe all text in reading order, exactly as printed (numbers, units, dates, signatures, stamps).
Render tables as Markdown tables. Skip the running header/footer and page number at the page edges.
Do not correct, summarise or add anything."""

SCAN_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "The main heading of the page."},
        "text": {"type": "string", "description": "Full transcription in Markdown."},
    },
    "required": ["title", "text"],
    "additionalProperties": False,
}


@dataclass
class Line:
    text: str
    bbox: tuple[float, float, float, float]
    size: float
    bold: bool


# ---------------------------------------------------------------- helpers


def doc_id_for(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")


def doc_title_for(doc: pymupdf.Document, path: Path) -> str:
    title = (doc.metadata or {}).get("title", "").strip()
    return title if title and title != "(unspecified)" else path.stem.replace("_", " ")


def _norm(text: str) -> str:
    return re.sub(r"\d+", "#", " ".join(text.lower().split()))


def _page_lines(page: pymupdf.Page) -> list[list[Line]]:
    """Text-layer blocks (roughly paragraphs) in reading order, each a list of lines."""
    blocks = []
    for block in page.get_text("dict", sort=True)["blocks"]:
        lines = []
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = " ".join("".join(s["text"] for s in spans).split())
            size = max(s["size"] for s in spans)
            bold = all(s["flags"] & 16 or "bold" in s["font"].lower() for s in spans)
            lines.append(Line(text, tuple(line["bbox"]), size, bold))
        if lines:
            blocks.append(lines)
    return blocks


def find_boilerplate(doc: pymupdf.Document) -> set[str]:
    """Lines in the top/bottom 10% of the page that repeat on at least half the pages (digits ignored)."""
    if len(doc) < 4:
        return set()
    counts: Counter[str] = Counter()
    for page in doc:
        h = page.rect.height
        seen = {
            _norm(line.text)
            for block in _page_lines(page)
            for line in block
            if line.bbox[3] < h * 0.1 or line.bbox[1] > h * 0.9
        }
        counts.update(seen)
    return {text for text, n in counts.items() if n >= len(doc) / 2}


def body_font_size(blocks_by_page: list[list[list[Line]]]) -> float:
    sizes: Counter[float] = Counter()
    for blocks in blocks_by_page:
        for block in blocks:
            for line in block:
                sizes[round(line.size, 1)] += len(line.text)
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _inside(bbox, rects, pad: float = 2.0) -> bool:
    x, y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return any(r.x0 - pad <= x <= r.x1 + pad and r.y0 - pad <= y <= r.y1 + pad for r in rects)


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text) if s.strip()]


def pack_paragraphs(paragraphs: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    """Greedily pack paragraphs into chunks of up to max_chars. Over-long paragraphs are split at sentence
    ends. Consecutive chunks share up to overlap_chars of trailing sentences so no fact is cut in half."""
    units: list[str] = []
    for p in paragraphs:
        units.extend([p] if len(p) <= max_chars else _sentences(p))
    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and len("\n".join(current + [unit])) > max_chars:
            chunks.append("\n".join(current))
            tail: list[str] = []
            for s in reversed(_sentences(" ".join(current))):
                if len(s) + sum(len(t) for t in tail) > overlap_chars:
                    break
                tail.insert(0, s)
            current = [" ".join(tail)] if tail else []
        current.append(unit)
    if current:
        chunks.append("\n".join(current))
    return chunks


def table_markdown(rows: list[list[str]]) -> str:
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _table_rows(page: pymupdf.Page, table) -> list[list[str]]:
    """Cell text read from each cell's box. (Table.extract() drops spaces and misplaces thousands
    separators on some PDFs, e.g. '6910 ,' for '6,910'.) The first row is the header."""
    rows = [
        [" ".join(page.get_textbox(pymupdf.Rect(cell)).split()) if cell else "" for cell in row.cells]
        for row in table.rows
    ]
    return [r for r in rows if any(r)]


def _image_regions(page: pymupdf.Page) -> list[pymupdf.Rect]:
    area = page.rect.get_area()
    regions: list[pymupdf.Rect] = []
    for img in page.get_images(full=True):
        for rect in page.get_image_rects(img[0]):
            rect = rect & page.rect
            if rect.get_area() / area >= MIN_IMAGE_AREA and not any(rect == r for r in regions):
                regions.append(rect)
    return regions


def _vector_figure_regions(page: pymupdf.Page, avoid: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    area = page.rect.get_area()
    regions = []
    for rect in page.cluster_drawings():
        frac = rect.get_area() / area
        if VECTOR_FIGURE_AREA <= frac <= 0.9 and not any(rect.intersects(a) for a in avoid):
            regions.append(rect)
    return regions


def _cached(cache_dir: Path, key_parts: list[bytes | str], call) -> dict:
    digest = hashlib.sha256()
    for part in key_parts:
        digest.update(part if isinstance(part, bytes) else part.encode("utf-8"))
    path = cache_dir / f"{digest.hexdigest()[:24]}.json"
    if path.exists():
        return json.loads(path.read_text("utf-8"))
    result = call()
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), "utf-8")
    return result


# ---------------------------------------------------------------- ingestion


class _Builder:
    """Collects chunks for one document and numbers them per page and kind."""

    def __init__(self, doc_id: str, title: str):
        self.doc_id, self.title = doc_id, title
        self.chunks: list[Chunk] = []
        self._counts: Counter[tuple[int, str]] = Counter()

    def add(self, page: int, kind: str, method: str, section: str, label: str, text: str) -> None:
        self._counts[(page, kind)] += 1
        chunk_id = f"{self.doc_id}:p{page}:{kind}{self._counts[(page, kind)]}"
        self.chunks.append(Chunk(chunk_id, self.doc_id, self.title, page, kind, method, section, label, text.strip()))

    def add_text(self, page: int, section: str, paragraphs: list[str], kind: str = "text",
                 method: str = "text-layer", label: str | None = None) -> None:
        for text in pack_paragraphs(paragraphs, settings.chunk_chars, settings.chunk_overlap_chars):
            self.add(page, kind, method, section, label if label is not None else section, text)


def ingest_pdf(path: str | Path, store: ChunkStore | None, use_vision: bool = True, log=print) -> dict:
    """Extract, chunk, embed and store one PDF. With store=None nothing is embedded or stored (dry run)."""
    path = Path(path)
    doc = pymupdf.open(path)
    doc_id, title = doc_id_for(path), doc_title_for(doc, path)
    pages_dir = settings.index_dir / "pages" / doc_id
    cache_dir = settings.index_dir / "cache" / "vision"
    pages_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    usage = llm.Usage()
    builder = _Builder(doc_id, title)
    boilerplate = find_boilerplate(doc)
    all_blocks = [
        [[ln for ln in block if _norm(ln.text) not in boilerplate] for block in _page_lines(page)]
        for page in doc
    ]
    all_blocks = [[b for b in blocks if b] for blocks in all_blocks]
    body_size = body_font_size(all_blocks)
    h1_size = body_size * 1.6

    report_pages = []
    top, sub = "", ""  # current section headings, carried across pages
    prev_table: tuple[list[str], str] | None = None  # (header row, caption) of the last table seen

    for page, blocks in zip(doc, all_blocks):
        pno = page.number + 1
        page.get_pixmap(dpi=settings.render_dpi).save(pages_dir / f"p{pno}.png")
        page_area = page.rect.get_area()
        page_report = {"page": pno, "text": 0, "tables": 0, "figures": 0, "scanned": False, "skipped": []}

        images = _image_regions(page)
        body_chars = sum(len(ln.text) for block in blocks for ln in block)
        biggest = max((r.get_area() / page_area for r in images), default=0)

        if biggest >= SCAN_IMAGE_AREA and body_chars < SCAN_MAX_TEXT:
            page_report["scanned"] = True
            if not use_vision:
                page_report["skipped"].append("scanned page (vision disabled)")
            else:
                png = page.get_pixmap(dpi=FIGURE_DPI).tobytes("png")
                result = _cached(cache_dir, [VISION_PROMPT_VERSION, "scan", settings.vision_model, png], lambda: llm.chat_json(
                    settings.vision_model, SCAN_SYSTEM, "Transcribe this scanned page.", "scan_transcription",
                    SCAN_SCHEMA, image_png=png, usage=usage))
                top, sub = result["title"].strip() or top, ""
                builder.add_text(pno, top, [p for p in result["text"].split("\n\n") if p.strip()],
                                 kind="scan", method="vision-model", label=f"Scanned page: {top}")
                page_report["text"] += 1
            report_pages.append(page_report)
            log(f"  page {pno:>3}: scanned" + (f", skipped: {page_report['skipped'][0]}" if page_report["skipped"] else ", transcribed by vision model"))
            continue

        # Tables first, so their text is not duplicated as prose.
        tables = [(t, rows) for t in page.find_tables(strategy="lines").tables
                  if t.col_count >= 2 and len(rows := _table_rows(page, t)) >= 2]
        table_rects = [pymupdf.Rect(t.bbox) for t, _ in tables]
        figures = images + _vector_figure_regions(page, table_rects + images)
        if figures and not use_vision:
            page_report["skipped"].append(f"{len(figures)} figure(s) (vision disabled)")

        all_lines = [ln for block in blocks for ln in block]
        used_captions: set[int] = set()
        items: list[tuple[float, str, object]] = []  # (y, kind, payload), processed top to bottom

        for (_, rows), rect in zip(tables, table_rects):
            above = [ln for ln in all_lines
                     if rect.y0 - 45 <= ln.bbox[1] and ln.bbox[3] <= rect.y0 + 8 and not _inside(ln.bbox, [rect])]
            above.sort(key=lambda ln: (not _CAPTION.match(ln.text), rect.y0 - ln.bbox[3]))
            caption = base = ""
            if above and (_CAPTION.match(above[0].text) or above[0].bold):
                caption = base = above[0].text
                used_captions.add(id(above[0]))
            elif prev_table and prev_table[1] and [c.lower() for c in rows[0]] == [c.lower() for c in prev_table[0]]:
                base = prev_table[1]  # same header row as the previous table: it continues onto this page
                caption = f"{base} (continued)"
            prev_table = (rows[0], base)
            items.append((rect.y0, "table", (caption or f"Table on page {pno}", rows)))

        if use_vision:
            for rect in figures:
                items.append((rect.y0, "figure", rect))

        for block in blocks:
            para: list[str] = []
            para_y = None
            for ln in block:
                if id(ln) in used_captions or _inside(ln.bbox, table_rects + figures):
                    continue
                if ln.bold and ln.size >= body_size * 1.2 and len(ln.text) < 120:
                    if para:
                        items.append((para_y, "para", " ".join(para)))
                        para, para_y = [], None
                    items.append((ln.bbox[1], "h1" if ln.size >= h1_size else "h2", ln.text))
                else:
                    para.append(ln.text)
                    para_y = ln.bbox[1] if para_y is None else para_y
            if para:
                items.append((para_y, "para", " ".join(para)))

        page_text = " ".join(p for _, k, p in items if k == "para")[:1200]
        buffer: list[str] = []

        def section() -> str:
            return f"{top} > {sub}" if top and sub else top or sub

        def flush() -> None:
            if buffer:
                builder.add_text(pno, section(), buffer)
                page_report["text"] += 1
                buffer.clear()

        for _, kind, payload in sorted(items, key=lambda it: it[0]):
            if kind in ("h1", "h2"):
                flush()
                if kind == "h1":
                    top, sub = payload, ""
                else:
                    sub = payload
            elif kind == "para":
                buffer.append(payload)
            elif kind == "table":
                flush()
                caption, rows = payload
                builder.add(pno, "table", "table-parser", section(), caption, f"{caption}\n{table_markdown(rows)}")
                page_report["tables"] += 1
            elif kind == "figure":
                flush()
                rect = payload + (-6, -6, 6, 6)
                png = page.get_pixmap(dpi=FIGURE_DPI, clip=rect & page.rect).tobytes("png")
                context = f"Section: {section() or '(none)'}\nPage text: {page_text or '(none)'}"
                result = _cached(cache_dir, [VISION_PROMPT_VERSION, "figure", settings.vision_model, png, context],
                                 lambda: llm.chat_json(settings.vision_model, FIGURE_SYSTEM,
                                                       f"Describe this figure.\n\n{context}", "figure_reading",
                                                       FIGURE_SCHEMA, image_png=png, usage=usage))
                content = result["content"].strip()
                if not result["informative"] and not content:
                    page_report["skipped"].append(f"decorative image: {result['title']}")
                    continue
                fig_title = result["title"].strip()
                label = fig_title if _CAPTION.match(fig_title) else f"{result['kind'].capitalize()}: {fig_title}"
                builder.add(pno, "figure", "vision-model", section(), label, f"{fig_title}\n{content}")
                page_report["figures"] += 1
        flush()
        report_pages.append(page_report)
        log(f"  page {pno:>3}: {page_report['text']} text, {page_report['tables']} table, "
            f"{page_report['figures']} figure{' (scanned)' if page_report['scanned'] else ''}"
            + (f"  skipped: {'; '.join(page_report['skipped'])}" if page_report["skipped"] else ""))

    chunks = builder.chunks
    if store is not None:
        vectors = llm.embed([c.context_text() for c in chunks], usage)
        store.replace_document(doc_id, chunks, vectors)

    report = {
        "doc_id": doc_id,
        "title": title,
        "source": str(path),
        "pages": len(doc),
        "chunks": len(chunks),
        "chunks_by_kind": dict(Counter(c.kind for c in chunks)),
        "removed_boilerplate": sorted(boilerplate),
        "usage": usage.summary(),
        "per_page": report_pages,
    }
    reports_dir = settings.index_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{doc_id}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), "utf-8")
    (reports_dir / f"{doc_id}.chunks.jsonl").write_text(
        "\n".join(json.dumps(c.__dict__, ensure_ascii=False) for c in chunks), "utf-8")
    return report
