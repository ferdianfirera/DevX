"""Runtime settings. Everything can be overridden with environment variables or a .env file."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # Page images, vision-model cache and the embedded Qdrant store live here.
    index_dir: Path = Path(_env("DOCQA_INDEX_DIR", str(ROOT / "index")))
    # Empty = embedded Qdrant on disk under index_dir. Set e.g. http://localhost:6333 for a server.
    qdrant_url: str = _env("QDRANT_URL", "")
    qdrant_api_key: str = _env("QDRANT_API_KEY", "")
    collection: str = _env("DOCQA_COLLECTION", "docqa_chunks")
    chat_model: str = _env("DOCQA_CHAT_MODEL", "gpt-4.1")
    vision_model: str = _env("DOCQA_VISION_MODEL", "gpt-4.1")
    embed_model: str = _env("DOCQA_EMBED_MODEL", "text-embedding-3-small")
    # Grades answers in the evaluation; a stronger model than the one being graded.
    judge_model: str = _env("DOCQA_JUDGE_MODEL", "gpt-4.1")
    # Number of chunks handed to the answer model. See README "Chunking and retrieval".
    top_k: int = int(_env("DOCQA_TOP_K", "6"))
    # Text chunk size in characters (~4 chars per token, so ~300 tokens) and overlap.
    chunk_chars: int = int(_env("DOCQA_CHUNK_CHARS", "1200"))
    chunk_overlap_chars: int = int(_env("DOCQA_CHUNK_OVERLAP", "200"))
    render_dpi: int = int(_env("DOCQA_RENDER_DPI", "150"))

    # USD per 1M tokens (input, output), used only for the cost report.
    # Check https://openai.com/api/pricing before relying on these numbers.
    prices: dict = field(default_factory=lambda: {
        "gpt-4.1": (2.00, 8.00),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1-nano": (0.10, 0.40),
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-5": (1.25, 10.00),
        "gpt-5-mini": (0.25, 2.00),
        "text-embedding-3-small": (0.02, 0.0),
        "text-embedding-3-large": (0.13, 0.0),
    })


settings = Settings()


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int = 0) -> float | None:
    price = settings.prices.get(model)
    if price is None:
        return None
    return (prompt_tokens * price[0] + completion_tokens * price[1]) / 1_000_000
