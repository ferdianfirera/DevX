import json

import pytest

from docqa.ingest import ingest_pdf, pack_paragraphs, table_markdown

from .conftest import REPORT


def test_short_paragraphs_share_one_chunk():
    assert pack_paragraphs(["First para.", "Second para."], max_chars=100, overlap_chars=20) == ["First para.\nSecond para."]


def test_long_text_splits_at_sentences_with_overlap():
    sentences = [f"Sentence number {i} has some words." for i in range(12)]
    chunks = pack_paragraphs([" ".join(sentences)], max_chars=120, overlap_chars=40)
    assert len(chunks) > 1
    assert all(len(c) <= 120 + 40 for c in chunks)
    # Every sentence survives, and each chunk after the first starts with the previous chunk's last sentence.
    assert all(any(s in c for c in chunks) for s in sentences)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.split("\n")[0] in prev


def test_table_markdown_pads_ragged_rows():
    md = table_markdown([["Metric", "2024", "2025"], ["Revenue", "7,655"]])
    assert md.splitlines() == ["| Metric | 2024 | 2025 |", "| --- | --- | --- |", "| Revenue | 7,655 |  |"]


@pytest.mark.skipif(not REPORT.exists(), reason="sample report not in data/")
def test_sample_report_text_and_tables(tmp_settings):
    report = ingest_pdf(REPORT, store=None, use_vision=False, log=lambda *_: None)
    chunks = [json.loads(line) for line in (tmp_settings.index_dir / "reports" / f"{report['doc_id']}.chunks.jsonl").open(encoding="utf-8")]
    tables = {c["label"]: c["text"] for c in chunks if c["kind"] == "table"}

    assert len(tables) == 6
    assert "| Revenue | 6,910 | 7,655 | 8,420 |" in tables["Table 1. Five-line financial summary"]
    # Table 2 runs over two pages; the second part is linked to the first caption.
    assert "| ID-CBT | Cibitung, Bekasi | Indonesia | Cold chain | 21,000 | 96% | 2019 |" in tables[
        "Table 2. Major facilities at 31 December 2025 (continued)"]
    # Running footer removed, table cells not duplicated as prose.
    assert not any("Fictional document for assessment use only" in c["text"] for c in chunks)
    assert not any("6,910" in c["text"] for c in chunks if c["kind"] == "text")
    # Chart pages and the scanned page are detected (and reported as skipped without vision).
    pages = {p["page"]: p for p in report["per_page"]}
    assert pages[13]["scanned"]
    assert all(pages[p]["skipped"] for p in (3, 5, 6, 7, 9))
    assert {c["page"] for c in chunks if "96 electric trucks" in c["text"]} == {2, 12}
