"""Streamlit UI: ask questions about the indexed reports and check every answer against its cited pages.

Run locally:  streamlit run streamlit_app.py
Deploy:       share.streamlit.io with main file streamlit_app.py (README: "Deploy to Streamlit Community Cloud")
"""

import contextlib
import hmac
import os
import sys
import threading

import streamlit as st

st.set_page_config(page_title="Kirana IR Assistant", page_icon=":material/query_stats:")

# A second Python (e.g. Anaconda) on PATH can start its own, older Streamlit instead of the project's.
if tuple(int(p) for p in st.__version__.split(".")[:2]) < (1, 67):
    st.error(
        f"This app needs Streamlit 1.67 or newer, but was started by Streamlit {st.__version__} from "
        f"`{sys.executable}`. Start it with the project's environment:\n\n"
        "- Windows: `.venv\\Scripts\\python.exe -m streamlit run streamlit_app.py`\n"
        "- macOS / Linux: `.venv/bin/python -m streamlit run streamlit_app.py`"
    )
    st.stop()

# On Streamlit Community Cloud the keys live in the app's Secrets settings, not in a .env file. Loading them
# here, before docqa reads its settings, turns root-level secrets into environment variables. Locally,
# without a .streamlit/secrets.toml, this does nothing and docqa reads .env as usual.
st.secrets.load_if_toml_exists()

from docqa.answer import answer_question  # noqa: E402
from docqa.config import settings  # noqa: E402
from docqa.store import MODES, ChunkStore  # noqa: E402

MODE_LABELS = {"hybrid": "Hybrid", "dense": "Dense", "bm25": "BM25"}
STATUS = {
    "answered": ("Answered", "green", ":material/check_circle:"),
    "partial": ("Partly answered", "orange", ":material/rule:"),
    "not_found": ("Not in the report", "gray", ":material/search_off:"),
}
EXAMPLES = [
    "What was revenue in the fourth quarter of 2025?",
    "How many electric trucks did Kirana add in 2025?",
    "How many trucks per day run between Cikarang and Surabaya?",
    "What dividend per share will Kirana pay for 2025?",
]


def md(text: str) -> str:
    """Escape the characters Streamlit Markdown would turn into LaTeX ($) or strikethrough (~)."""
    return text.replace("$", r"\$").replace("~", r"\~")


def require_password() -> None:
    """If APP_PASSWORD is set, ask for it once per session. Keeps a public deployment from spending your API credit."""
    expected = os.getenv("APP_PASSWORD", "")
    if not expected or st.session_state.get("authenticated"):
        return
    st.title("Kirana IR Assistant")
    entered = st.text_input("Password", type="password")
    if entered and hmac.compare_digest(entered.encode(), expected.encode()):
        st.session_state.authenticated = True
        st.rerun()
    if entered:
        st.error("Incorrect password.")
    st.stop()


@st.cache_resource
def get_store() -> tuple[ChunkStore, contextlib.AbstractContextManager]:
    # One client shared by all sessions. The on-disk store allows one caller at a time; a Qdrant server does not need the lock.
    lock = contextlib.nullcontext() if settings.qdrant_url else threading.Lock()
    return ChunkStore(), lock


@st.cache_data(ttl=60, show_spinner=False)
def list_documents() -> list[dict]:
    store, lock = get_store()
    with lock:
        return store.documents()


def render_answer(a: dict, show_retrieved: bool) -> None:
    label, color, icon = STATUS.get(a["status"], (a["status"], "gray", None))
    st.badge(label, color=color, icon=icon)
    st.markdown(md(a["answer"]))
    if a["conflict"]:
        st.warning(f"**The sources disagree.** {md(a['conflict'])}", icon=":material/warning:")
    for warning in a["warnings"]:
        st.warning(md(warning), icon=":material/error:")

    for c in a["citations"]:
        with st.expander(md(f"[{c['n']}] {c['doc_title']}, {c['source']}"), icon=":material/description:"):
            excerpt = c["excerpt"] + (" …" if len(c["excerpt"]) >= 600 else "")
            st.markdown(md(excerpt))
            page_png = settings.index_dir / "pages" / c["doc_id"] / f"p{c['page']}.png"
            if page_png.is_file():
                st.image(str(page_png), caption=f"Page {c['page']}", width="stretch")

    if show_retrieved and a["retrieved"]:
        with st.expander(f"Retrieved excerpts ({len(a['retrieved'])})", icon=":material/manage_search:"):
            st.dataframe(a["retrieved"], hide_index=True, column_order=["n", "page", "kind", "score", "chunk_id"])

    cost = a["usage"].get("cost_usd")
    st.caption(f"{len(a['retrieved'])} excerpts · {a['latency_s']:.1f} s" + (f" · \\${cost:.4f}" if cost is not None else ""))


require_password()
store, lock = get_store()
try:
    docs = list_documents()
except Exception as e:  # wrong URL or key, cluster asleep, network
    st.error(f"Could not reach the vector store ({type(e).__name__}). Check QDRANT_URL and QDRANT_API_KEY.\n\n{e}")
    st.stop()

history: list[dict] = st.session_state.setdefault("history", [])

with st.sidebar:
    st.header("Settings")
    doc_titles = {"": "All documents"} | {d["doc_id"]: d["title"] for d in docs}
    doc_id = st.selectbox("Document", list(doc_titles), format_func=doc_titles.get)
    mode = st.segmented_control(
        "Retrieval", MODES, default="hybrid", required=True, format_func=MODE_LABELS.get,
        help="Hybrid fuses meaning-based (dense) and exact-word (BM25) search.",
    )
    k = st.slider("Excerpts given to the model", 1, 12, settings.top_k)
    show_retrieved = st.toggle("Show retrieved excerpts")
    if st.button("Clear conversation", icon=":material/delete:", width="stretch", disabled=not history):
        history.clear()
        st.rerun()
    st.divider()
    for d in docs:
        st.caption(f"**{d['title']}**  \n{d['pages']} pages · {d['chunks']} chunks")
    st.caption(f"Answers: `{settings.chat_model}`  \nEmbeddings: `{settings.embed_model}`")

st.title("Kirana IR Assistant")
st.caption("Answers come only from the indexed reports, with page citations. Each question is answered on its own.")

if not docs:
    st.warning("No documents are indexed yet. Run `python -m docqa ingest <file.pdf>` "
               "with the same QDRANT_URL as this app. The list refreshes every minute.", icon=":material/inbox:")
    st.stop()

for turn in history:
    with st.chat_message("user"):
        st.markdown(md(turn["question"]))
    with st.chat_message("assistant"):
        render_answer(turn["answer"], show_retrieved)

example = None
if not history:
    st.markdown("**Try a question**")
    with st.container(horizontal=True):
        for q in EXAMPLES:
            if st.button(q):
                example = q

question = st.chat_input("Ask about the report", max_chars=1000) or example
if question and question.strip():
    question = question.strip()
    with st.chat_message("user"):
        st.markdown(md(question))
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the report and writing the answer…"), lock:
                answer = answer_question(store, question, k=k, mode=mode, doc_id=doc_id or None).to_dict()
        except Exception as e:
            st.error(f"Could not answer the question ({type(e).__name__}): {e}")
        else:
            history.append({"question": question, "answer": answer})
            st.rerun()  # redraw from history so the example buttons disappear
