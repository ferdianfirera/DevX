"""Command line: ingest PDFs, ask questions, run the evaluation, start the HTTP API."""

import argparse
import json
import sys
import textwrap

from .config import settings
from .store import MODES


def _print_answer(ans, show_context: bool = False) -> None:
    print(f"\n{ans.status.upper()}")
    print(textwrap.fill(ans.answer, width=100))
    if ans.conflict:
        print("\nConflict in the sources: " + textwrap.fill(ans.conflict, width=100))
    if ans.citations:
        print("\nSources:")
        for c in ans.citations:
            print(f"  [{c.n}] {c.doc_title}, {c.source}")
    for w in ans.warnings:
        print(f"\nWarning: {w}")
    cost = ans.usage.get("cost_usd")
    print(f"\n({len(ans.retrieved)} excerpts retrieved, {ans.latency_s:.1f} s"
          + (f", ${cost:.4f}" if cost is not None else "") + ")")
    if show_context:
        print("\nRetrieved:")
        for r in ans.retrieved:
            print(f"  [{r['n']}] p.{r['page']} {r['kind']:<6} {r['chunk_id']}  score={r['score']}")


def cmd_ingest(args) -> None:
    from .ingest import ingest_pdf
    from .store import ChunkStore

    store = None if args.dry_run else ChunkStore()
    try:
        for pdf in args.pdfs:
            print(f"Ingesting {pdf}")
            report = ingest_pdf(pdf, store, use_vision=not args.no_vision)
            print(f"-> {report['doc_id']}: {report['chunks']} chunks {report['chunks_by_kind']}")
            print(f"   API usage: {report['usage']}")
            print(f"   Chunks written to {settings.index_dir / 'reports' / (report['doc_id'] + '.chunks.jsonl')}")
    finally:
        if store is not None:
            store.close()


def cmd_ask(args) -> None:
    from .answer import answer_question
    from .store import ChunkStore

    store = ChunkStore()

    def run(question: str) -> None:
        ans = answer_question(store, question, k=args.k, mode=args.mode, doc_id=args.doc)
        if args.json:
            print(json.dumps(ans.to_dict(), indent=2, ensure_ascii=False))
        else:
            _print_answer(ans, args.show_context)

    try:
        if args.question:
            run(args.question)
        else:
            while question := input("\nQuestion (blank to quit)> ").strip():
                run(question)
    finally:
        store.close()


def cmd_docs(args) -> None:
    from .store import ChunkStore

    store = ChunkStore()
    try:
        docs = store.documents()
    finally:
        store.close()
    if not docs:
        print("No documents indexed yet. Run: python -m docqa ingest <file.pdf>")
    for d in docs:
        print(f"{d['doc_id']}  ({d['title']}, {d['pages']} pages, {d['chunks']} chunks)")


def cmd_eval(args) -> None:
    from .evaluate import run_evaluation

    run_evaluation(args.questions, modes=args.modes.split(","), k=args.k, out_dir=args.out)


def cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run("docqa.server:app", host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to a legacy code page

    parser = argparse.ArgumentParser(prog="python -m docqa", description="Question answering over PDF reports.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Extract, chunk and index one or more PDFs (re-ingesting replaces them).")
    p.add_argument("pdfs", nargs="+")
    p.add_argument("--no-vision", action="store_true", help="Skip charts, images and scanned pages.")
    p.add_argument("--dry-run", action="store_true", help="Extract and chunk only; no embeddings, nothing stored.")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="Ask a question (omit it for an interactive session).")
    p.add_argument("question", nargs="?")
    p.add_argument("--mode", choices=MODES, default="hybrid")
    p.add_argument("--k", type=int, default=settings.top_k)
    p.add_argument("--doc", help="Restrict to one doc_id (see `docs`).")
    p.add_argument("--json", action="store_true")
    p.add_argument("--show-context", action="store_true", help="Also list every retrieved excerpt.")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("docs", help="List indexed documents.")
    p.set_defaults(func=cmd_docs)

    p = sub.add_parser("eval", help="Run the evaluation set and write a report.")
    p.add_argument("--questions", default="eval/questions.jsonl")
    p.add_argument("--modes", default="hybrid,dense,bm25", help="Comma-separated retrieval modes to compare.")
    p.add_argument("--k", type=int, default=settings.top_k)
    p.add_argument("--out", default="eval/results")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("serve", help="Start the HTTP API (the UI is: streamlit run streamlit_app.py).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)
