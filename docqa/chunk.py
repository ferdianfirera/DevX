"""The unit of retrieval: a piece of one page, with enough metadata to cite it."""

from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str   # "<doc_id>:p<page>:<kind><n>", stable across re-ingestion of the same file
    doc_id: str
    doc_title: str
    page: int       # 1-based, as printed in the PDF viewer
    kind: str       # "text" | "table" | "figure" | "scan"
    method: str     # how the content was obtained: "text-layer" | "table-parser" | "vision-model"
    section: str    # nearest heading above the chunk, "" if none
    label: str      # human-readable source, e.g. "Table: Revenue by country"
    text: str

    def context_text(self) -> str:
        """What gets embedded and keyword-indexed: the content plus where it sits in the document."""
        label = self.label if self.label != self.section else ""
        where = " | ".join(p for p in (self.doc_title, f"page {self.page}", self.section, label) if p)
        return f"{where}\n{self.text}"

    def citation(self) -> str:
        source = f"p. {self.page}, {self.label}" if self.label else f"p. {self.page}"
        if self.method == "vision-model":
            source += " (read from image)"
        return source
