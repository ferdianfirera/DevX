"""Qdrant chunk store with dense (embedding), sparse (BM25) and hybrid (RRF-fused) search.

Each chunk is one Qdrant point with two named vectors:
  - "dense": an OpenAI embedding of the chunk's context text (semantic match)
  - "bm25":  a sparse term-weight vector; Qdrant applies IDF at query time (exact words, numbers, names)
Hybrid search runs both as prefetches and lets Qdrant fuse them with Reciprocal Rank Fusion.
"""

import re
import uuid
import zlib
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from . import llm
from .chunk import Chunk
from .config import settings

DENSE = "dense"
SPARSE = "bm25"
MODES = ("hybrid", "dense", "bm25")

# BM25 parameters. Like Qdrant's own BM25 (fastembed), we use a fixed average length instead of the
# corpus average, so adding a document never requires re-weighting the existing ones.
K1, B, AVG_LEN = 1.2, 0.75, 200
PREFETCH = 20  # candidates taken from each side before fusion

# Keeps "1,234.5" and "12.3" as single tokens so figures can be matched exactly.
_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be by did do does for from had has have how in is it its of on or "
    "that the this to was were what when where which who why with".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def _token_id(token: str) -> int:
    return zlib.crc32(token.encode("utf-8"))


def bm25_doc_vector(text: str) -> models.SparseVector:
    tokens = tokenize(text)
    length_norm = 1 - B + B * len(tokens) / AVG_LEN
    weights = {_token_id(t): n * (K1 + 1) / (n + K1 * length_norm) for t, n in Counter(tokens).items()}
    return models.SparseVector(indices=list(weights), values=list(weights.values()))


def bm25_query_vector(text: str) -> models.SparseVector:
    ids = sorted({_token_id(t) for t in tokenize(text)})
    return models.SparseVector(indices=ids, values=[1.0] * len(ids))


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _doc_filter(doc_id: str | None) -> models.Filter | None:
    if not doc_id:
        return None
    return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])


def _chunk(payload: dict[str, Any] | None) -> Chunk:
    # Every point is stored with its chunk as payload, so None means it was fetched without one.
    if payload is None:
        raise ValueError("Qdrant point has no payload; query it with with_payload=True")
    return Chunk(**payload)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


class ChunkStore:
    def __init__(self) -> None:
        if settings.qdrant_url:
            self.qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        else:
            # Embedded mode: data on local disk, no server. One process at a time can open it.
            self.qdrant = QdrantClient(path=str(settings.index_dir / "qdrant"))
        self.collection = settings.collection

    def close(self) -> None:
        self.qdrant.close()

    def _ensure_collection(self, dim: int) -> None:
        if self.qdrant.collection_exists(self.collection):
            return
        self.qdrant.create_collection(
            self.collection,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        if settings.qdrant_url:  # payload indexes only matter on a server
            self.qdrant.create_payload_index(self.collection, "doc_id", models.PayloadSchemaType.KEYWORD)

    def replace_document(self, doc_id: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Drop any earlier version of this document, then store the new chunks."""
        self._ensure_collection(vectors.shape[1])
        self.qdrant.delete(self.collection, points_selector=models.FilterSelector(filter=_doc_filter(doc_id)))
        points = [
            models.PointStruct(
                id=_point_id(c.chunk_id),
                vector={DENSE: v.tolist(), SPARSE: bm25_doc_vector(c.context_text())},
                payload=asdict(c),
            )
            for c, v in zip(chunks, vectors)
        ]
        self.qdrant.upsert(self.collection, points=points)

    def all_chunks(self, doc_id: str | None = None) -> list[Chunk]:
        if not self.qdrant.collection_exists(self.collection):
            return []
        chunks, offset = [], None
        while True:
            points, offset = self.qdrant.scroll(
                self.collection, scroll_filter=_doc_filter(doc_id), limit=256, offset=offset, with_payload=True
            )
            chunks.extend(_chunk(p.payload) for p in points)
            if offset is None:
                break
        return sorted(chunks, key=lambda c: (c.doc_id, c.page, c.chunk_id))

    def documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for c in self.all_chunks():
            d = docs.setdefault(c.doc_id, {"doc_id": c.doc_id, "title": c.doc_title, "pages": 0, "chunks": 0})
            d["pages"] = max(d["pages"], c.page)
            d["chunks"] += 1
        return list(docs.values())

    def search(
        self,
        query: str,
        k: int,
        mode: str = "hybrid",
        doc_id: str | None = None,
        usage: llm.Usage | None = None,
    ) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not self.qdrant.collection_exists(self.collection):
            return []
        flt = _doc_filter(doc_id)
        sparse = bm25_query_vector(query)
        dense = llm.embed([query], usage)[0].tolist() if mode != "bm25" else None

        if mode == "dense":
            res = self.qdrant.query_points(self.collection, query=dense, using=DENSE, query_filter=flt, limit=k)
        elif mode == "bm25":
            if not sparse.indices:
                return []
            res = self.qdrant.query_points(self.collection, query=sparse, using=SPARSE, query_filter=flt, limit=k)
        else:
            prefetch = [models.Prefetch(query=dense, using=DENSE, filter=flt, limit=PREFETCH)]
            if sparse.indices:
                prefetch.append(models.Prefetch(query=sparse, using=SPARSE, filter=flt, limit=PREFETCH))
            res = self.qdrant.query_points(
                self.collection, prefetch=prefetch, query=models.FusionQuery(fusion=models.Fusion.RRF), limit=k
            )
        return [Hit(_chunk(p.payload), p.score, rank) for rank, p in enumerate(res.points, start=1)]
