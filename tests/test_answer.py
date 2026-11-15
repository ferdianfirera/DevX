from docqa import answer as answer_mod
from docqa.chunk import Chunk
from docqa.store import Hit


class FakeStore:
    def __init__(self, hits):
        self.hits = hits

    def search(self, *args, **kwargs):
        return self.hits


def _hits():
    chunks = [
        Chunk("d:p4:table1", "d", "Report", 4, "table", "table-parser", "Financial review", "Table 1", "EBITDA 1,186"),
        Chunk("d:p5:figure1", "d", "Report", 5, "figure", "vision-model", "Revenue", "Figure 1", "Q4: 2,305"),
    ]
    return [Hit(c, 0.5, rank) for rank, c in enumerate(chunks, start=1)]


def _patch_reply(monkeypatch, reply):
    monkeypatch.setattr(answer_mod.llm, "chat_json", lambda *a, **k: reply)


def test_only_supplied_excerpts_become_citations(monkeypatch):
    _patch_reply(monkeypatch, {"status": "answered", "answer": "EBITDA was THB 1,186 million [1][7].",
                               "citations": [1, 7], "conflict": None})
    ans = answer_mod.answer_question(FakeStore(_hits()), "EBITDA 2024?")
    assert [(c.n, c.page) for c in ans.citations] == [(1, 4)]
    assert any("[7]" in w for w in ans.warnings)


def test_inline_markers_count_as_citations(monkeypatch):
    _patch_reply(monkeypatch, {"status": "answered", "answer": "Q4 revenue was THB 2,305 million [2].",
                               "citations": [], "conflict": None})
    ans = answer_mod.answer_question(FakeStore(_hits()), "Q4 revenue?")
    assert [c.page for c in ans.citations] == [5]
    assert "(read from image)" in ans.citations[0].source


def test_sources_cited_in_the_conflict_note_become_citations(monkeypatch):
    _patch_reply(monkeypatch, {"status": "answered", "answer": "EBITDA was THB 1,186 million [1].", "citations": [1],
                               "conflict": "The chart on page 5 gives a different figure [2]."})
    ans = answer_mod.answer_question(FakeStore(_hits()), "EBITDA 2024?")
    assert [c.page for c in ans.citations] == [4, 5]


def test_answer_without_citation_is_flagged(monkeypatch):
    _patch_reply(monkeypatch, {"status": "answered", "answer": "About 2,000.", "citations": [], "conflict": None})
    ans = answer_mod.answer_question(FakeStore(_hits()), "Revenue?")
    assert ans.citations == []
    assert any("unverified" in w for w in ans.warnings)


def test_refusal_needs_no_citation(monkeypatch):
    _patch_reply(monkeypatch, {"status": "not_found", "answer": "The report does not mention dividends.",
                               "citations": [], "conflict": None})
    ans = answer_mod.answer_question(FakeStore(_hits()), "Dividend per share?")
    assert ans.status == "not_found" and ans.warnings == []


def test_empty_index_refuses_without_calling_the_model(monkeypatch):
    monkeypatch.setattr(answer_mod.llm, "chat_json", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    ans = answer_mod.answer_question(FakeStore([]), "Anything?")
    assert ans.status == "not_found"
