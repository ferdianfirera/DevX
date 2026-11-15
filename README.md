# Kirana IR Assistant

Answers investor-relations questions about PDF company reports, using only what the report says. Every
answer cites the page it came from. When the report doesn't contain the answer, the assistant says so
instead of guessing.

The sample document is `Kirana_Logistics_Annual_Report_2025.pdf`, a 13-page report with prose, tables,
charts, a diagram and a scanned page.

```
PDF ─► ingest ─────────────────────────────► chunks ─► embed + BM25 ─► Qdrant
        prose:   PDF text layer                         text-embedding-3-small
        tables:  table parser → Markdown
        charts, diagrams, scanned pages: gpt-4.1 vision

question ─► hybrid search (dense + BM25, RRF) ─► top 6 chunks ─► gpt-4.1 ─► answer + verified citations
                                                                               │
                                                CLI · Streamlit UI · HTTP API · evaluation report
```

Model calls go through LangChain (`langchain-openai`).

---

## Run the full pipeline

All commands run from the `kirana-ir-assistant` folder.

### 1. Prerequisites

- **Python 3.12 or newer.** The pinned `numpy` needs 3.12; tested with Python 3.14.
- **An OpenAI API key** with access to `gpt-4.1` and `text-embedding-3-small`.
- **A vector store.** Pick one:
  - **Qdrant Cloud** (recommended). Create a free cluster at <https://cloud.qdrant.io>, then copy its
    URL and API key.
  - **A local Qdrant server.** Run `docker run -p 6333:6333 qdrant/qdrant`.
  - **Nothing.** Leave `QDRANT_URL` empty and the app stores everything on disk under `index/qdrant`.
    Only one process can open that store at a time (see [Troubleshooting](#troubleshooting)).

### 2. Install

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure

Copy the template and fill in your keys:

```bash
cp env.example .env        # Windows: copy env.example .env
```

```ini
OPENAI_API_KEY=sk-...

# Qdrant Cloud or a local server. Leave both empty for the on-disk store.
QDRANT_URL=https://<cluster-id>.<region>.cloud.qdrant.io:6333
QDRANT_API_KEY=...
```

`.env` is ignored by git. Every other setting has a sensible default (see
[Configuration](#configuration)).

### 4. Add the report

PDFs in `data/` are ignored by git, so copy the report in yourself:

```
data/Kirana_Logistics_Annual_Report_2025.pdf
```

### 5. Check the setup (optional, free)

```bash
python -m pytest
```

The tests make no OpenAI calls and never touch your Qdrant cluster. They use fake embeddings and a
throwaway local store. Expect `14 passed`. LangChain prints a few harmless `DeprecationWarning`s on
Python 3.14.

To preview how the PDF will be split, without any API calls or writes to Qdrant:

```bash
python -m docqa ingest data/Kirana_Logistics_Annual_Report_2025.pdf --dry-run --no-vision
```

Use both flags. `--dry-run` alone still sends charts to the vision model.

### 6. Ingest the report

```bash
python -m docqa ingest data/Kirana_Logistics_Annual_Report_2025.pdf
```

This takes under a minute and costs about **$0.03** (7 charts and diagrams plus the scanned page read by
gpt-4.1, then one embedding call):

```
  page   1: 2 text, 0 table, 1 figure
  ...
  page  13: scanned, transcribed by vision model
-> kirana-logistics-annual-report-2025: 34 chunks {'text': 20, 'figure': 7, 'table': 6, 'scan': 1}
   API usage: {'calls': 9, 'prompt_tokens': 15877, 'completion_tokens': 1095, 'seconds': 20.997, 'cost_usd': 0.031921}
```

Confirm that it's indexed:

```bash
python -m docqa docs
# kirana-logistics-annual-report-2025  (Kirana Logistics Group Annual Report 2025, 13 pages, 34 chunks)
```

Ingestion writes:

| Where | What |
| --- | --- |
| Qdrant collection `docqa_chunks` | One point per chunk, with a dense vector, a BM25 vector and the chunk text |
| `index/pages/<doc_id>/p<N>.png` | Rendered pages, shown under each citation in the Streamlit app |
| `index/cache/vision/` | Vision-model results. Re-ingesting the same PDF doesn't pay for them again |
| `index/reports/<doc_id>.json` | Per-page report: chunks found, figures skipped, API usage |
| `index/reports/<doc_id>.chunks.jsonl` | Every chunk exactly as stored, for inspection |

Ingesting the same file again replaces its chunks instead of duplicating them. To add more reports,
pass several paths: `python -m docqa ingest a.pdf b.pdf`.

### 7. Ask questions

One question:

```bash
python -m docqa ask "What was revenue in the fourth quarter of 2025?"
```

```
ANSWERED
Revenue in the fourth quarter of 2025 was THB 2,305 million.

Sources:
  [2] Kirana Logistics Group Annual Report 2025, p. 5, Figure 1. Revenue by quarter, 2025 (THB million) (read from image)

(6 excerpts retrieved, 2.8 s, $0.0027)
```

Leave out the question to start an interactive session (press Enter on a blank line to quit). Options:

| Flag | Effect |
| --- | --- |
| `--mode hybrid\|dense\|bm25` | Retrieval method (default `hybrid`) |
| `--k 6` | Number of excerpts given to the model |
| `--doc <doc_id>` | Search one document only (ids from `python -m docqa docs`) |
| `--show-context` | Also list every retrieved excerpt with its page and score |
| `--json` | Print the full answer object as JSON |

Each answer has a status:

- `answered`: the report answers the question.
- `partial`: the report covers only part of the question.
- `not_found`: the report doesn't contain the answer.

If two sources in the report give different values, the answer shows both instead of picking one.

### 8. Web UI (Streamlit)

Start it with the project's Python, so the Streamlit version from `requirements.txt` is used:

```powershell
.venv\Scripts\python.exe -m streamlit run streamlit_app.py      # Windows
```

```bash
.venv/bin/python -m streamlit run streamlit_app.py              # macOS / Linux
```

Open <http://localhost:8501>.

With the virtual environment active, `streamlit run streamlit_app.py` works too. If Anaconda or
another Python comes first on your PATH, though, that command can start an older Streamlit (see
[Troubleshooting](#troubleshooting)).

- **Asking.** Type a question or click one of the example questions. Each question is answered on its
  own; earlier ones aren't used as context.
- **Each answer shows:**
  - its status (answered, partly answered, or not in the report)
  - a note when sources disagree
  - one expandable source per citation, with the excerpt and the rendered page so you can check the
    answer by eye
  - the number of excerpts, the latency and the cost
- **Sidebar.** Pick the document, the retrieval mode (hybrid, dense or BM25) and how many excerpts the
  model gets. You can also show the retrieved excerpts with their scores, or clear the conversation.

To put the app online, see [Deploy to Streamlit Community Cloud](#deploy-to-streamlit-community-cloud).

### 9. HTTP API (optional)

To call the assistant from other programs, start the FastAPI server:

```bash
python -m docqa serve            # --host 0.0.0.0 --port 8000 to change
```

Interactive API docs are at <http://127.0.0.1:8000/docs>, and the root URL redirects there.

| Endpoint | Purpose |
| --- | --- |
| `POST /api/ask` | Body `{"question": "...", "doc_id": null, "mode": "hybrid"}`. Returns the answer, citations, retrieved chunks, usage and latency |
| `GET /api/docs` | Indexed documents |
| `GET /pages/{doc_id}/{page}.png` | Rendered page image |

```bash
curl -s -X POST http://127.0.0.1:8000/api/ask -H "Content-Type: application/json" \
  -d '{"question": "What was EBITDA in 2024?"}'
```

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/ask -ContentType "application/json" `
  -Body '{"question": "What was EBITDA in 2024?"}'
```

### 10. Evaluate

```bash
python -m docqa eval
```

This runs the 20 questions in [`eval/questions.jsonl`](eval/questions.jsonl) through each retrieval mode
and grades every answer with a judge model:

- 16 answerable: prose, tables, charts, an infographic, a diagram, the scanned page, and one figure
  that the CEO letter states and an errata later corrects.
- 4 the report can't answer, which should get a refusal.

The default run makes about 60 answer calls and 60 judge calls on gpt-4.1. Expect it to take a few minutes
and cost well under $1. Add `--modes hybrid` to run one mode only.

Results go to `eval/results/`:

| File | Contents |
| --- | --- |
| `report.md` | Summary table per mode, scores by content type, every question, and each failure with the likely stage (retrieval or answer) |
| `summary.json` | The same metrics, machine-readable |
| `<mode>.jsonl` | Full record per question: answer, retrieved and cited pages, judge verdict and reason |

Metrics:

| Kind | Metric | Meaning |
| --- | --- | --- |
| Retrieval | Hit@k | At least one expected page was retrieved |
| Retrieval | Page recall | Share of expected pages retrieved |
| Retrieval | MRR | 1 ÷ rank of the first chunk from an expected page |
| Answers | Answer score | Correct = 1, partially correct = 0.5, incorrect = 0 |
| Answers | False refusals | Said "not found" when the answer was in the report |
| Answers | Citation accuracy | A cited page is one of the expected pages |
| Answers | Correct refusals | Refused, as it should, on the 4 unanswerable questions |

Other options: `--k`, `--questions <file>`, `--out <dir>`.

---

## Deploy to Streamlit Community Cloud

The deployed app only answers questions. Ingestion stays on your machine and writes to Qdrant Cloud,
which the app reads from.

1. **Ingest into Qdrant Cloud first.** With `QDRANT_URL` set in your local `.env`, run step 6. The
   on-disk store won't work on Streamlit Cloud: its files aren't kept, and `index/` isn't in git.

2. **Push this folder to GitHub.** The repository can be public or private. `.gitignore` already keeps
   these out of git:
   - `.env` and `.streamlit/secrets.toml`
   - `index/`
   - `data/*.pdf`

3. **Create the app.** At <https://share.streamlit.io>, choose **Create app**, then pick the repository
   and branch.
   - **Main file path:** `streamlit_app.py`. Use `kirana-ir-assistant/streamlit_app.py` if this folder
     isn't the repository root.
   - **Python version:** under **Advanced settings**, choose **Python 3.12 or newer**.
     `requirements.txt` needs it.

4. **Add the secrets.** Also under **Advanced settings**, paste them into **Secrets** in TOML format.
   The quotes are required:

   ```toml
   OPENAI_API_KEY = "sk-..."
   QDRANT_URL = "https://<cluster-id>.<region>.cloud.qdrant.io:6333"
   QDRANT_API_KEY = "..."
   APP_PASSWORD = "choose-a-password"   # optional, see below
   ```

   Streamlit turns these into environment variables, so the app reads them the same way it reads `.env`
   locally. You can add any other setting from [Configuration](#configuration) here too.

5. **Deploy.** Every push to the branch redeploys the app.

**Protect your API credit.** Anyone with the link can ask questions, and each one is paid for by your
OpenAI key. Each question costs about $0.003 with gpt-4.1. Do one or both:

- Set `APP_PASSWORD`. The app then asks for it once per browser session.
- Make the app private in its **Share** settings, and invite viewers by email.

A monthly budget on your OpenAI project also caps the damage.

**Page images.** `index/pages/` isn't in git, so the deployed app shows each citation's excerpt without
the page picture. To include the pictures:

- In `.gitignore`, replace `index/` with these two lines:
  ```
  index/*
  !index/pages/
  ```
- Commit `index/pages/`.

Only do this if the report may be published. The images show every page in full.

---

## Configuration

All settings are environment variables, read from `.env`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | (required) | OpenAI access |
| `QDRANT_URL` | empty: on-disk store | Qdrant server or cloud cluster URL |
| `QDRANT_API_KEY` | empty | Qdrant Cloud API key |
| `DOCQA_COLLECTION` | `docqa_chunks` | Qdrant collection name |
| `DOCQA_CHAT_MODEL` | `gpt-4.1` | Writes the answers |
| `DOCQA_VISION_MODEL` | `gpt-4.1` | Reads charts, diagrams and scanned pages during ingestion |
| `DOCQA_JUDGE_MODEL` | `gpt-4.1` | Grades answers in the evaluation |
| `DOCQA_EMBED_MODEL` | `text-embedding-3-small` | Embeds chunks and questions |
| `DOCQA_TOP_K` | `6` | Excerpts given to the answer model |
| `DOCQA_CHUNK_CHARS` | `1200` | Maximum prose chunk length in characters (~300 tokens) |
| `DOCQA_CHUNK_OVERLAP` | `200` | Characters of trailing sentences repeated in the next chunk |
| `DOCQA_RENDER_DPI` | `150` | Resolution of the page images shown in the UI |
| `DOCQA_INDEX_DIR` | `index/` | Page images, vision cache, reports and the on-disk store |
| `APP_PASSWORD` | empty: no password | If set, the Streamlit app asks for it before answering |

Streamlit also reads `.streamlit/secrets.toml` (same keys, TOML format) if you prefer it to `.env`
locally. That file is ignored by git too.

Changing the chunk settings or vision model takes effect on the next ingest. Re-run ingest afterwards.

---

## How it works

### Ingestion

Each page is handled by the method that suits its content:

| Content | Method | Chunk |
| --- | --- | --- |
| Prose | PDF text layer (PyMuPDF), split at headings | Packed paragraphs. Never crosses a page |
| Tables | PyMuPDF table finder, cells read box by box | One Markdown table per chunk, with its caption. A table that continues onto the next page is linked to the first caption |
| Charts, infographics, diagrams | Region cropped at 200 dpi and read by the vision model | One chunk per figure, every printed value as `label: value` |
| Scanned pages | Whole page transcribed by the vision model | Treated like prose |

Running headers and footers are detected (lines that repeat on half the pages) and removed. Decorative
images are skipped. Every chunk keeps its page number for citation.

### Chunking and retrieval

- **Chunk size.** Prose chunks are up to 1,200 characters (about 300 tokens). Long paragraphs are split
  at sentence ends, and the next chunk repeats up to 200 characters of trailing sentences, so no fact is
  cut in half.
- **What gets indexed.** Each chunk is indexed together with its location (document title, page,
  section heading and label). An excerpt like "Q4: 2,305" is then still found by a question about
  "revenue by quarter".
- **Hybrid search.** Each chunk gets two vectors in Qdrant:
  - **Dense:** a `text-embedding-3-small` embedding, which matches meaning.
  - **BM25:** keyword weights, with Qdrant applying IDF at query time. The tokenizer keeps figures like
    `1,234.5` as single tokens, so exact numbers, names and years match.

  Each side returns 20 candidates. Qdrant merges them with Reciprocal Rank Fusion, and the top
  `DOCQA_TOP_K` (6) go to the answer model.
- **Comparing settings.** `dense` and `bm25` can each be run alone. Compare all three with
  `python -m docqa eval` before changing `DOCQA_TOP_K` or the chunk size.

### Answering

gpt-4.1 (temperature 0, JSON-schema structured output) gets the question and the numbered excerpts, with
these rules:

- Use only the excerpts, and cite every statement.
- Return `not_found` or `partial` instead of guessing.
- Report conflicting values instead of picking one.
- Say when a figure was read from an image.

The code then checks the citations:

- A citation that points at an excerpt the model wasn't given is dropped, with a warning.
- An answer with no valid citation is flagged as unverified.

---

## Project layout

```
kirana-ir-assistant/
├── streamlit_app.py       Streamlit UI (local and Streamlit Community Cloud)
├── data/                  PDFs to ingest (not in git)
├── docqa/
│   ├── cli.py             python -m docqa {ingest, ask, docs, eval, serve}
│   ├── config.py          settings from .env, model prices for the cost report
│   ├── ingest.py          PDF → chunks (text layer, tables, vision)
│   ├── chunk.py           Chunk record and citation format
│   ├── llm.py             LangChain: ChatOpenAI (structured JSON) and OpenAIEmbeddings, usage and cost
│   ├── store.py           Qdrant collection, BM25 vectors, dense / bm25 / hybrid search
│   ├── answer.py          grounded answer and citation check
│   ├── evaluate.py        retrieval and answer metrics, report.md
│   └── server.py          FastAPI HTTP API
├── eval/questions.jsonl   20 evaluation questions with expected answers and pages
├── index/                 generated by ingest (not in git)
├── tests/                 pytest, no API calls
├── env.example
└── requirements.txt
```

---

## Troubleshooting

**`No documents are indexed yet`.** Run step 6. If you did, check that `QDRANT_URL` and
`DOCQA_COLLECTION` match the store you ingested into. Without `QDRANT_URL` the app reads the on-disk
store, not your cloud cluster.

**`Storage folder ... is already accessed by another instance of Qdrant client`.** The on-disk store
allows one process at a time. Stop the Streamlit app or `serve` before running `ask` or `eval`, or use
a Qdrant server or cloud cluster.

**`TypeError: ... got an unexpected keyword argument`, or `This app needs Streamlit 1.67 or newer`.**
The app was started by another Python's Streamlit, for example Anaconda's. The path in the message
shows which one. Start it with `.venv\Scripts\python.exe -m streamlit run streamlit_app.py`. In VS Code,
also run **Python: Select Interpreter** and choose `.venv`, so new terminals use the project's
environment.

**The Streamlit app says `Could not reach the vector store`.** Check `QDRANT_URL` and `QDRANT_API_KEY`
in `.env`, or in the app's Secrets on Streamlit Cloud. Free Qdrant Cloud clusters can be suspended
when idle. If yours is, resume it in the Qdrant console.

**The Streamlit app says `No documents are indexed yet`.** It reads the same `QDRANT_URL` and
`DOCQA_COLLECTION` that you ingested into. A new ingest shows up within a minute.

**Dimension mismatch after changing `DOCQA_EMBED_MODEL`.** The collection's vector size is fixed when it
is first created (1,536 for `text-embedding-3-small`, 3,072 for `text-embedding-3-large`). Set a new
`DOCQA_COLLECTION` name, or delete the old collection, then re-ingest.

**`401` / authentication errors.** Check `OPENAI_API_KEY` or `QDRANT_API_KEY` in `.env`. Shell
environment variables with the same name take precedence over `.env`.

**Costs show as `None`.** The model isn't in the price table in `docqa/config.py`. Add it there.
