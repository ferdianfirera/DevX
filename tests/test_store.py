import numpy as np

from docqa import llm
from docqa.chunk import Chunk
from docqa.store import ChunkStore, bm25_doc_vector, tokenize


def _chunk(doc, page, text, kind="text"):
    return Chunk(f"{doc}:p{page}:{kind}1", doc, doc.title(), page, kind, "text-layer", "", "", text)


def test_tokenize_keeps_figures_and_drops_stopwords():
    assert tokenize("What was the revenue of THB 1,234.5 million in 2025?") == ["revenue", "thb", "1,234.5", "million", "2025"]


def test_bm25_term_frequency_saturates():
    one = dict(zip(*[bm25_doc_vector("truck").indices, bm25_doc_vector("truck").values]))
    many = bm25_doc_vector("truck " * 20)
    weight_many = dict(zip(many.indices, many.values))
    key = next(iter(one))
    assert weight_many[key] > one[key]
    assert weight_many[key] < 2.2 * one[key]  # bounded by k1 + 1, unlike raw counts


def test_hybrid_search_filters_and_replaces(tmp_settings):
    store = ChunkStore()
    try:
        chunks = [
            _chunk("report-a", 4, "EBITDA was THB 1,186 million in 2024."),
            _chunk("report-a", 9, "The electric fleet grew to 218 vehicles."),
            _chunk("report-b", 2, "EBITDA guidance for next year."),
        ]
        for doc in ("report-a", "report-b"):
            mine = [c for c in chunks if c.doc_id == doc]
            store.replace_document(doc, mine, llm.embed([c.context_text() for c in mine]))

        for mode in ("hybrid", "dense", "bm25"):
            hits = store.search("electric fleet vehicles", k=2, mode=mode)
            assert hits[0].chunk.page == 9, mode
        assert {h.chunk.doc_id for h in store.search("EBITDA", k=5, doc_id="report-b")} == {"report-b"}

        # Re-ingesting a document replaces its chunks instead of duplicating them.
        new = [_chunk("report-a", 1, "Replacement text only.")]
        store.replace_document("report-a", new, llm.embed([new[0].context_text()]))
        assert [c.page for c in store.all_chunks("report-a")] == [1]
        assert {d["doc_id"] for d in store.documents()} == {"report-a", "report-b"}
    finally:
        store.close()


def test_search_on_empty_store_returns_nothing(tmp_settings):
    store = ChunkStore()
    try:
        assert store.search("anything", k=3) == []
    finally:
        store.close()
