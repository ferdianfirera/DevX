import dataclasses
import hashlib

import numpy as np
import pytest

from docqa import config, llm, store

REPORT = config.ROOT / "data" / "Kirana_Logistics_Annual_Report_2025.pdf"


def fake_embed(texts, usage=None, batch_size=64):
    """Deterministic bag-of-words embedding, so retrieval tests need no API key."""
    vecs = np.zeros((len(texts), 256), dtype=np.float32)
    for i, text in enumerate(texts):
        for token in store.tokenize(text):
            vecs[i, int(hashlib.md5(token.encode()).hexdigest(), 16) % 256] += 1
    vecs += 1e-6
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@pytest.fixture
def tmp_settings(tmp_path, monkeypatch):
    """Point every module at a throwaway index directory with an embedded Qdrant."""
    s = dataclasses.replace(config.settings, index_dir=tmp_path, qdrant_url="", collection="test_chunks")
    for module in ("docqa.config", "docqa.store", "docqa.ingest", "docqa.answer"):
        monkeypatch.setattr(f"{module}.settings", s)
    monkeypatch.setattr(llm, "embed", fake_embed)
    return s
