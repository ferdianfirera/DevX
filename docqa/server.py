"""HTTP API for other programs. The user interface is the Streamlit app (streamlit_app.py).

  GET  /          redirects to the interactive API docs at /docs
  POST /api/ask   {"question": "...", "doc_id": null, "mode": "hybrid"} -> answer with citations
  GET  /api/docs  indexed documents
  GET  /pages/{doc_id}/{page}.png  rendered page image, so a citation can be checked by eye
"""

import re
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

from .answer import answer_question
from .config import settings
from .store import MODES, ChunkStore

_store: ChunkStore | None = None
_lock = threading.Lock()  # embedded Qdrant is not built for concurrent access


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _store
    _store = ChunkStore()
    yield
    _store.close()


app = FastAPI(title="Report Q&A", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    doc_id: str | None = None
    mode: str = "hybrid"


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    question = req.question.strip()
    if not question or len(question) > 1000:
        raise HTTPException(400, "Ask a question of 1 to 1000 characters.")
    if req.mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    with _lock:
        return answer_question(_store, question, mode=req.mode, doc_id=req.doc_id or None).to_dict()


@app.get("/api/docs")
def docs() -> list[dict]:
    with _lock:
        return _store.documents()


@app.get("/pages/{doc_id}/{page}.png")
def page_image(doc_id: str, page: int) -> FileResponse:
    if not re.fullmatch(r"[a-z0-9-]+", doc_id):
        raise HTTPException(404)
    path = settings.index_dir / "pages" / doc_id / f"p{page}.png"
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse("/docs")
