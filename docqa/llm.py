"""Model layer on LangChain (langchain-openai): embeddings and structured (JSON-schema) chat, with or
without an image. The rest of the package only uses embed(), chat_json() and Usage from here."""

import base64
import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import tiktoken
from langchain_core.exceptions import ModelInvalidRequestError, ModelRefusalError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import cost_usd, settings

# Models that rejected temperature=0 (reasoning models); we stop sending it to them.
_no_temperature: set[str] = set()


@lru_cache(maxsize=None)
def _chat_model(model: str, zero_temperature: bool) -> ChatOpenAI:
    # Reads OPENAI_API_KEY from the environment (config loads .env).
    return ChatOpenAI(model=model, **({"temperature": 0} if zero_temperature else {}))


@lru_cache(maxsize=None)
def _embeddings(model: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model)


@lru_cache(maxsize=None)
def _encoding(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


@dataclass
class Usage:
    """Token, latency and cost accounting for one question or one ingestion run."""
    calls: list[dict] = field(default_factory=list)

    def add(self, model: str, prompt_tokens: int, completion_tokens: int, seconds: float) -> None:
        self.calls.append({
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "seconds": round(seconds, 3),
            "cost_usd": cost_usd(model, prompt_tokens, completion_tokens),
        })

    def summary(self) -> dict:
        costs = [c["cost_usd"] for c in self.calls]
        return {
            "calls": len(self.calls),
            "prompt_tokens": sum(c["prompt_tokens"] for c in self.calls),
            "completion_tokens": sum(c["completion_tokens"] for c in self.calls),
            "seconds": round(sum(c["seconds"] for c in self.calls), 3),
            "cost_usd": None if None in costs else round(sum(costs), 6),
        }


def embed(texts: list[str], usage: Usage | None = None, batch_size: int = 64) -> np.ndarray:
    """Return L2-normalised embeddings, one row per text."""
    model = settings.embed_model
    rows: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        t0 = time.perf_counter()
        rows.extend(_embeddings(model).embed_documents(batch))
        seconds = time.perf_counter() - t0
        if usage is not None:
            # LangChain does not return token usage for embeddings. It sends tiktoken token ids to the
            # API, so counting with the same tokenizer gives the billed number.
            usage.add(model, sum(len(_encoding(model).encode(t)) for t in batch), 0, seconds)
    vecs = np.asarray(rows, dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def chat_json(
    model: str,
    system: str,
    user_text: str,
    schema_name: str,
    schema: dict,
    image_png: bytes | None = None,
    usage: Usage | None = None,
) -> dict:
    """Ask the model for a reply that must validate against `schema` (OpenAI structured outputs)."""
    content: list[dict] = [{"type": "text", "text": user_text}]
    if image_png is not None:
        b64 = base64.b64encode(image_png).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}})
    messages = [SystemMessage(system), HumanMessage(content=content)]

    def invoke(zero_temperature: bool) -> dict:
        # include_raw keeps the AIMessage so token usage can be recorded; parse errors and refusals
        # come back in "parsing_error" instead of being raised.
        structured = _chat_model(model, zero_temperature).with_structured_output(
            {"name": schema_name, "strict": True, "schema": schema}, method="json_schema", include_raw=True
        )
        return structured.invoke(messages)

    t0 = time.perf_counter()
    try:
        out = invoke(model not in _no_temperature)  # temperature 0: repeatable answers for evaluation
    except ModelInvalidRequestError as e:
        if model in _no_temperature or "temperature" not in str(e):
            raise
        _no_temperature.add(model)
        out = invoke(False)
    if usage is not None:
        meta = out["raw"].usage_metadata or {}
        usage.add(model, meta.get("input_tokens", 0), meta.get("output_tokens", 0), time.perf_counter() - t0)

    if (err := out["parsing_error"]) is not None:
        if isinstance(err, ModelRefusalError):
            raise RuntimeError(f"Model refused: {err}") from err
        raise err
    return out["parsed"]
