"""Evaluation: retrieval and answer quality are measured separately, for each retrieval mode.

Retrieval (answerable questions only), from the pages of the top-k chunks:
  hit@k        at least one expected page was retrieved
  page recall  share of the expected pages that were retrieved (matters when evidence spans pages)
  MRR          1 / rank of the first chunk from an expected page
Answers, graded by a judge model against the reference answer:
  correct / partially correct / incorrect, plus correct refusals on unanswerable questions,
  false refusals, and citation accuracy (a cited page is one of the expected pages).
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

from . import llm
from .answer import answer_question
from .config import settings
from .store import ChunkStore

JUDGE_SYSTEM = """You grade an assistant that answers questions about a company report, against a reference
answer written by someone who read the whole report.
- correct: gives the reference facts (numbers, names, periods) accurately. Extra correct detail and different
  wording are fine.
- partially_correct: some reference facts right, but something important is missing or slightly wrong.
- incorrect: wrong facts, invented facts, or a refusal when the reference has an answer.
When the reference says the report does not contain the answer: correct only if the assistant says the report
does not contain it, without inventing or inferring a figure; incorrect if it supplies an answer anyway.
Grade factual content only, not style."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partially_correct", "incorrect"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}

SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


def load_questions(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def judge(q: dict, ans) -> dict:
    prompt = (
        f"Question: {q['question']}\n"
        f"Reference answer: {q['expected_answer']}\n"
        f"Report contains the answer: {'yes' if q['answerable'] else 'no'}\n\n"
        f"Assistant status: {ans.status}\n"
        f"Assistant answer: {ans.answer}\n"
        f"Assistant conflict note: {ans.conflict or '(none)'}"
    )
    return llm.chat_json(settings.judge_model, JUDGE_SYSTEM, prompt, "grade", JUDGE_SCHEMA)


def evaluate_one(store: ChunkStore, q: dict, mode: str, k: int) -> dict:
    ans = answer_question(store, q["question"], k=k, mode=mode)
    expected = set(q["expected_pages"])
    pages = [r["page"] for r in ans.retrieved]
    cited = sorted({c.page for c in ans.citations})

    retrieval = None
    if q["answerable"]:
        first = next((rank for rank, p in enumerate(pages, start=1) if p in expected), None)
        retrieval = {
            "hit": first is not None,
            "rr": 1 / first if first else 0.0,
            "page_recall": len(expected & set(pages)) / len(expected),
        }
    grade = judge(q, ans)
    return {
        "id": q["id"],
        "type": q["type"],
        "answerable": q["answerable"],
        "question": q["question"],
        "expected_answer": q["expected_answer"],
        "expected_pages": q["expected_pages"],
        "status": ans.status,
        "answer": ans.answer,
        "conflict": ans.conflict,
        "retrieved_pages": pages,
        "retrieved": ans.retrieved,
        "cited_pages": cited,
        "citation_ok": bool(expected & set(cited)) if q["answerable"] else None,
        "retrieval": retrieval,
        "verdict": grade["verdict"],
        "judge_reason": grade["reason"],
        "warnings": ans.warnings,
        "latency_s": ans.latency_s,
        "cost_usd": ans.usage.get("cost_usd"),
        "tokens": ans.usage.get("prompt_tokens", 0) + ans.usage.get("completion_tokens", 0),
    }


def summarise(results: list[dict]) -> dict:
    ans = [r for r in results if r["answerable"]]
    unans = [r for r in results if not r["answerable"]]
    costs = [r["cost_usd"] for r in results if r["cost_usd"] is not None]
    return {
        "questions": len(results),
        "hit_at_k": sum(r["retrieval"]["hit"] for r in ans) / len(ans),
        "mrr": statistics.mean(r["retrieval"]["rr"] for r in ans),
        "page_recall": statistics.mean(r["retrieval"]["page_recall"] for r in ans),
        "answer_score": statistics.mean(SCORE[r["verdict"]] for r in ans),
        "fully_correct": sum(r["verdict"] == "correct" for r in ans),
        "answerable": len(ans),
        "false_refusals": sum(r["status"] == "not_found" for r in ans),
        "citation_accuracy": sum(bool(r["citation_ok"]) for r in ans) / len(ans),
        "correct_refusals": sum(r["verdict"] == "correct" for r in unans),
        "unanswerable": len(unans),
        "avg_latency_s": statistics.mean(r["latency_s"] for r in results),
        "p95_latency_s": sorted(r["latency_s"] for r in results)[int(0.95 * (len(results) - 1))],
        "avg_cost_usd": statistics.mean(costs) if costs else None,
        "avg_tokens": statistics.mean(r["tokens"] for r in results),
    }


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def write_report(path: Path, summaries: dict[str, dict], results: dict[str, list[dict]], primary: str, k: int) -> None:
    lines = [
        "# Evaluation results",
        "",
        f"{summaries[primary]['questions']} questions ({summaries[primary]['answerable']} answerable, "
        f"{summaries[primary]['unanswerable']} not answerable from the report). top-k = {k}. "
        f"Answer model `{settings.chat_model}`, vision model `{settings.vision_model}`, "
        f"embeddings `{settings.embed_model}`, judge `{settings.judge_model}`.",
        "",
        "## Summary by retrieval mode",
        "",
        "| Mode | Hit@k | Page recall | MRR | Answer score | Fully correct | False refusals | Citation accuracy | Correct refusals | Avg latency | Avg cost |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for mode, s in summaries.items():
        cost = f"${s['avg_cost_usd']:.4f}" if s["avg_cost_usd"] is not None else "n/a"
        lines.append(
            f"| {mode} | {_pct(s['hit_at_k'])} | {_pct(s['page_recall'])} | {s['mrr']:.2f} | {_pct(s['answer_score'])} | "
            f"{s['fully_correct']}/{s['answerable']} | {s['false_refusals']} | {_pct(s['citation_accuracy'])} | "
            f"{s['correct_refusals']}/{s['unanswerable']} | {s['avg_latency_s']:.1f} s | {cost} |"
        )

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results[primary]:
        by_type[r["type"]].append(r)
    lines += ["", f"## By content type ({primary})", "", "| Type | Questions | Retrieved right page | Answer score |", "| --- | --- | --- | --- |"]
    for t, rs in by_type.items():
        hit = [r["retrieval"]["hit"] for r in rs if r["retrieval"]]
        lines.append(f"| {t} | {len(rs)} | {f'{sum(hit)}/{len(hit)}' if hit else 'n/a'} | "
                     f"{_pct(statistics.mean(SCORE[r['verdict']] for r in rs))} |")

    lines += ["", f"## Every question ({primary})", "",
              "| ID | Type | Expected pages | Retrieved pages | Cited | Status | Verdict |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in results[primary]:
        lines.append(f"| {r['id']} | {r['type']} | {r['expected_pages'] or '-'} | {r['retrieved_pages']} | "
                     f"{r['cited_pages'] or '-'} | {r['status']} | {r['verdict']} |")

    failures = [r for r in results[primary] if r["verdict"] != "correct"]
    lines += ["", f"## Failures ({primary})", ""]
    if not failures:
        lines.append("None.")
    for r in failures:
        if r["retrieval"] and not r["retrieval"]["hit"]:
            stage = "retrieval: no chunk from an expected page reached the model"
        elif r["retrieval"] and r["retrieval"]["page_recall"] < 1:
            stage = "retrieval: only part of the evidence was retrieved"
        else:
            stage = "answer generation or extraction: the right page was retrieved"
        lines += [
            f"### {r['id']} ({r['type']}): {r['question']}",
            f"- Expected: {r['expected_answer']}",
            f"- Got ({r['status']}): {r['answer']}",
            f"- Judge: {r['verdict']}: {r['judge_reason']}",
            f"- Likely stage: {stage}",
            "",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(questions_path: str, modes: list[str], k: int, out_dir: str) -> dict:
    questions = load_questions(questions_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    store = ChunkStore()
    results: dict[str, list[dict]] = {}
    try:
        for mode in modes:
            print(f"\n== {mode} ==")
            results[mode] = []
            for q in questions:
                r = evaluate_one(store, q, mode, k)
                results[mode].append(r)
                hit = "-" if r["retrieval"] is None else ("hit " if r["retrieval"]["hit"] else "MISS")
                print(f"  {r['id']} {r['type']:<12} retrieval {hit}  {r['status']:<9} {r['verdict']}")
            with open(out / f"{mode}.jsonl", "w", encoding="utf-8") as f:
                for r in results[mode]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    finally:
        store.close()

    summaries = {mode: summarise(rs) for mode, rs in results.items()}
    (out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    write_report(out / "report.md", summaries, results, modes[0], k)
    print(f"\nReport written to {out / 'report.md'}")
    for mode, s in summaries.items():
        print(f"  {mode:<7} hit@k {_pct(s['hit_at_k'])}  answer score {_pct(s['answer_score'])}  "
              f"correct refusals {s['correct_refusals']}/{s['unanswerable']}")
    return summaries
