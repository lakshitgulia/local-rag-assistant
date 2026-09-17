# Local RAG Assistant

A fully local retrieval-augmented question-answering system: point it at a folder of
documents, ask questions in plain English, get answers grounded in cited excerpts from
those documents — no cloud APIs, no data leaving the machine. Built originally as a
client-bid proof of concept, then extended into an always-on personal document
assistant that watches a folder and stays current automatically.

## What it does

1. **Ingest** — parses PDF, PPTX, DOCX, EML, MBOX, and scanned images (OCR fallback)
   from a folder, splits the text into overlapping chunks, and embeds each chunk with a
   local Ollama embedding model.
2. **Index** — stores chunk vectors in a FAISS index and chunk metadata (source file,
   page/slide/location, role) in `index/chunks.json`.
3. **Retrieve** — for each question, runs hybrid search: FAISS vector similarity *and*
   BM25 keyword search over the same chunks, fused with weighted Reciprocal Rank Fusion.
4. **Generate** — hands the top-ranked excerpts to a local Ollama LLM with a prompt that
   forbids outside knowledge and forbids blending unrelated excerpts together, and asks
   for a cited, thorough answer.
5. **Refuse honestly** — if nothing relevant is retrieved, the system says so instead of
   guessing (the guardrail).

## Architecture

```
core.py         Shared engine primitives: config, embeddings/LLM factories, role
                assignment, tokenizer, and JSON persistence (chunks/manifest/watcher status)
ingest.py       Parsing, chunking, incremental embedding + FAISS indexing
retrieval.py    Hybrid retrieval (EnsembleRetriever), guardrail, citation building, LCEL generation chain
watcher.py      Background filesystem watcher — calls ingest.py's incremental logic on file changes
app.py          FastAPI server: /api/query, /api/roles, /api/stats, /api/ingest/status, static UI
static/index.html   Single-page chat + ingestion-status UI, wired only to real endpoints above
eval.py         Citation-accuracy eval harness (question -> expected source file)
eval_questions.json  15 real eval questions drawn from indexed content
notebooks/      5 numbered walkthrough notebooks calling the real engine on real data
index/          Persisted FAISS store, chunks.json, manifest.json, watcher_status.json (gitignored)
```

The email connector originally scoped for this project (IMAP pull, safety-gated,
never auto-run) was built during an early phase and later removed entirely at the
user's direction — there is no email code anywhere in the current codebase.

## Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Orchestration | LangChain 1.4 (`langchain`, `langchain-core`, `langchain-community`, `langchain-classic`, `langchain-ollama`) | See "Built with LangChain" below for why each package is there |
| Vector store | FAISS (`faiss-cpu`) | Local, on-disk, no external service |
| Keyword search | BM25 (`langchain_community.retrievers.BM25Retriever`, backed by `rank_bm25`) | Lazy-imported by that retriever at query time — kept as an explicit dependency even though nothing imports it directly |
| Embeddings | `nomic-embed-text` via Ollama | 274MB, local |
| LLM | `llama3.2:3b` via Ollama (default) | See below — `llama3.1:8b` is still installed but no longer the default |
| API | FastAPI + Pydantic | `/api/query`, `/api/roles`, `/api/stats`, `/api/ingest/status` |
| Document parsing | `pypdf`, `python-pptx`, `python-docx`, `pytesseract` + `pdf2image` (OCR fallback) | |
| Background watcher | `watchdog` | Real filesystem events, not polling |
| Demo notebooks | `jupyter`/`notebook` | Dev-only, not imported by the running app |

**On the LLM default — what actually happened:** the project started with `llama3.1:8b`
(4.9GB) as the default. On the 8GB-RAM development machine, that model caused swap
thrashing under load, pushing individual query latency to 100–320 seconds. After
benchmarking alternatives, `llama3.2:3b` (2.0GB) became the default — it responds in
single-digit-to-low-double-digit seconds on the same hardware. **`llama3.1:8b` was not
removed** — it's still pulled and available locally (`ollama list` shows both models) —
it's just no longer what `.env.example` points to by default. Swap `OLLAMA_LLM_MODEL` in
`.env` to switch back on a machine with more RAM headroom.

Switching models surfaced a real quality regression (see Bug 5 below), fixed with a
prompt change before the switch was finalized.

### Built with LangChain — why each package is there

- `langchain` / `langchain-core`: the base framework — `Document`, `Embeddings`,
  `VectorStore`, `PromptTemplate`, LCEL (`prompt | llm | parser`).
- `langchain-community`: the FAISS vectorstore wrapper and `BM25Retriever`.
- `langchain-ollama`: `OllamaEmbeddings` and `OllamaLLM`, the local-model bindings.
- `langchain-classic`: `EnsembleRetriever` — LangChain 1.x split legacy composability
  retrievers out of the main package into this one.
- `langsmith`: an **unavoidable transitive dependency of `langchain-core`**, not
  something this project chose to add. It was verified inert: with no `LANGCHAIN_API_KEY`
  set, `lsof` during live queries showed connections only to `127.0.0.1:11434` (Ollama) —
  nothing reaches LangSmith's servers.
- `langgraph`: an unavoidable transitive dependency of `langchain` itself. Unused —
  no agents or graphs are built in this project.

LangChain's own `Embeddings` / `VectorStore` / `LLM` base classes already are the
swappable-provider abstraction this project needs (see the docstring in `core.py`) —
there's no separate hand-rolled abstraction layer on top of them.

## Getting your documents in

Drop files into the folder set by `LOCAL_SCAN_DIR` in `.env` (defaults to `~/Downloads`),
then run:

```bash
python ingest.py
```

Supported types: `.pdf`, `.pptx`, `.docx`, `.eml`, `.mbox`, plus scanned images
(`.png`/`.jpg`/`.jpeg`/`.tiff`/`.bmp`) via OCR fallback. Ingestion is **incremental**:
each file's SHA-256 hash is recorded in `index/manifest.json`, so unchanged files are
skipped on every subsequent run, and a changed file has its old chunks removed from the
FAISS index before the new ones are added — nothing is ever double-indexed.

## Always-on setup (personal-use mode)

Beyond the one-shot `python ingest.py`, the project can run continuously in the
background so newly downloaded documents become searchable without a manual step:

- **`watcher.py`** uses `watchdog` to monitor `LOCAL_SCAN_DIR` for real filesystem
  create/modify events (not polling), debounces each path by 3 seconds so a
  still-downloading file isn't ingested mid-write, and calls the same incremental
  `run_ingestion()` logic `ingest.py` uses — no duplicated ingestion code. Run it with
  `python watcher.py`.
- **macOS LaunchAgents** keep both the API server and the watcher running persistently
  and restart them if they crash:
  - `~/Library/LaunchAgents/com.localragassistant.server.plist` — runs
    `uvicorn app:app --host 0.0.0.0 --port 8000` via the venv's Python directly
    (invoking the `uvicorn` entry-point script through `launchd`'s shell fallback failed
    with a working-directory error — calling `python3 -u -m uvicorn` avoids that).
  - `~/Library/LaunchAgents/com.localragassistant.watcher.plist` — runs `watcher.py`.
  - Both set `RunAtLoad` + `KeepAlive` (auto-start on login, auto-restart on crash) and
    log to `~/Library/Logs/localragassistant-*.log`.
  - Load with `launchctl load ~/Library/LaunchAgents/com.localragassistant.<name>.plist`;
    check they're running with `launchctl list | grep localragassistant`.
- The **Document Ingestion** tab in the web UI (and the `/api/ingest/status` endpoint it
  calls) shows live index stats and the watcher's last-processed file/timestamp, read
  fresh from `index/manifest.json` and `index/watcher_status.json` on every request —
  not cached, so it reflects changes the watcher makes even though it's a separate
  process from the API server.

This is personal-use scaffolding, not a production deployment: see **Known
Limitations** below for what it doesn't handle (deleted-file pruning, scale, etc.).

## Run it

```bash
./run.sh
```

This pulls any missing Ollama models named in `.env`, creates/activates a venv,
installs `requirements.txt`, runs `ingest.py` once if no index exists yet, and starts
the API + UI at `http://localhost:8000`.

## Role-based access (demo stub)

A file under a folder literally named `finance` (case-insensitive) is tagged
finance-only; everything else is visible to `all`. Pass `role=finance` to `/api/query`
to see the difference. This is a folder-name convention for demo purposes, **not** real
access control — a production system would derive roles from the source system's own
permissions.

## How hybrid retrieval + citations work

1. `retriever.invoke(question)` runs BM25 and FAISS vector search in parallel via
   LangChain's `EnsembleRetriever`, fusing results with weighted Reciprocal Rank Fusion
   (`c=60`, equal 0.5/0.5 weights). This was verified to implement the same algorithm as
   the project's original hand-written fusion by reading `EnsembleRetriever`'s source —
   not assumed — before letting it replace the hand-written version outright.
2. Results are filtered by role visibility, and the top 5 are handed to the LLM as
   numbered excerpts.
3. The prompt instructs the model to answer using *only* those excerpts, to cite them
   inline as `[1]`, `[2]`, etc., and explicitly warns that excerpts may come from
   different source documents and must not be blended together.
4. If retrieval returns nothing relevant, the system returns a fixed refusal instead of
   calling the LLM at all (the guardrail) — checked via a substring marker on the
   response rather than an exact-string match, since small local models tend to
   paraphrase a fixed refusal sentence.
5. The top 3 retrieved chunks are returned to the caller as numbered source citations
   (file, location, snippet) — see **Known Limitations** for a gap between this and what
   the model is allowed to cite inline.

## Web UI

`static/index.html` is a single-page app served at `/`, wired only to the real
endpoints listed above — nothing in it is mocked or hardcoded:

- **Chat** — ask a question, pick a role, see the answer with numbered citation cards
  (file-type badges, source file, location, snippet), a real "Grounded in N sources" /
  "No relevant source found" indicator, and the actual response time in milliseconds.
  A "Strict grounding" toggle is shown permanently on with an explanatory caption — see
  **Known Limitations**. Feedback buttons on each answer are present but, on click,
  honestly disclose that nothing is saved yet (no feedback endpoint exists).
- **Document Ingestion** — live stats from `/api/ingest/status`: total documents,
  total chunks, a breakdown by file type, when the index was last updated, and the
  watcher's last processed file, with a manual refresh button and an honest "not yet
  run" empty state before the watcher has done anything.

## Bug history

Real bugs found and fixed during development, kept here (with the historical rationale
comments still in the source) because they document *why* the code looks the way it
does, not just what it does:

1. **Word-based chunking crash.** A PDF with broken text extraction produced glyphs
   with no spaces between them, which word-based chunking turned into a single
   ~4,638-token "word" that blew past Ollama's embedding batch limit and crashed
   ingestion. Fixed by switching to character-based chunking (`ingest.py`,
   `CHUNK_CHARS = 2000`), which can't produce an unbounded token regardless of how
   extraction breaks.
2. **Naive BM25 tokenization.** BM25Retriever's default tokenizer is a plain
   `str.split()`, so `"AutoMind?"` and `"automind"` were treated as different tokens,
   hurting keyword recall. Fixed with an explicit regex tokenizer (`core.py:tokenize`)
   passed in as `preprocess_func`.
3. **Guardrail exact-string match failed on paraphrase.** The refusal check originally
   compared the model's output to the fixed refusal sentence exactly, but small local
   models paraphrase it (e.g. inserting the topic into the sentence). Fixed by checking
   for a substring marker instead (`NO_ANSWER_MARKER`).
4. **Thin answers, then a timeout regression from fixing them.** Early answers were
   one-line summaries because Ollama's unset default generation length is small. Fixed
   by setting `num_predict=700` and strengthening the prompt to ask for thorough,
   detail-citing answers. That fix then caused occasional raw 500 errors: a full
   700-token answer sometimes took longer than the client's default request timeout.
   Fixed by raising `sync_client_kwargs={"timeout": 600}` — a direct, minimal
   consequence of the first fix, not unrelated scope creep.
5. **Cross-document content blending after the model switch.** Benchmarking
   `llama3.2:3b` as a faster replacement for `llama3.1:8b` surfaced a real quality
   regression: the smaller model sometimes blended facts from two unrelated retrieved
   excerpts into one answer as if they were the same source. Fixed by adding an explicit
   prompt instruction that numbered excerpts may come from different documents and must
   not be blended — then re-tested the specific failing question plus the other two
   benchmark questions before finalizing the model switch.
6. **LangChain migration batching crash.** `OllamaEmbeddings.embed_documents()` sends
   its entire input list in one HTTP request; handing it 882+ chunks at once during the
   LangChain migration crashed Ollama's `/api/embed` with a connection reset after
   about 30 seconds. Fixed with client-side batching at `EMBED_BATCH_SIZE = 64`
   (`ingest.py`).

## Measured accuracy

`eval.py` runs 15 real questions (drawn from actual indexed content, 3 of them
deliberately adversarial — asking about things not in the corpus, or phrased to bait a
guess) against `/api/query` and checks whether the expected source file appears in the
top-3 returned citations.

**Result: 13/15 correct (87%).**

Both misses were investigated rather than shrugged off:
- One was a genuinely ambiguous question where two different indexed documents both
  contained plausible partial answers — the retriever returned a reasonable but
  non-expected file in position 1–3.
- The other was a near-miss where the expected file was retrieved but ranked just
  outside the top-3 citation cutoff (`CITATION_K = 3`) despite being in the top-5
  context window the LLM actually saw.

All 3 adversarial questions passed — the guardrail correctly refused to answer when
nothing relevant was indexed.

Re-run anytime with:
```bash
python eval.py eval_questions.json
```

**2026-09-17 update — corpus drift caused a temporary drop to 10/15, root-caused and
resolved for the demo corpus.** Between the original eval and this date, the watcher
auto-ingested 15 files that were drafts of this project's own DPR and Summer Training
Report (by the project's author) sitting in `~/Downloads` — legitimate watcher behavior,
not a bug. These near-duplicate, self-referential documents out-ranked the real expected
source on 4 of the 15 questions, dropping the score to 10/15 (67%). Root cause confirmed
by checking that a code simplification pass made the same day touched none of the
retrieval path, and by tracing every wrong citation back to one of the 15 drafts. Fixed
for the demo corpus by moving all 15 files out of the watched folder entirely (not
deleted — kept at `~/rag_excluded_from_demo`, a location *outside* `LOCAL_SCAN_DIR` on
purpose: a first attempt that moved them into a subfolder *of* `~/Downloads` didn't work,
because the watcher scans its target folder recursively, so any "excluded" folder must
live outside the watched tree, not inside it) and manually pruning their chunks from
the index — `watcher.py`/`ingest.py` don't auto-prune on file removal (see Known
Limitations), so this was a one-off manual step, not a new automated feature. Score
returned to 13/15 (87%), matching the original baseline exactly. The underlying gap this
exposed — no automated drift detection when new documents dilute retrieval for existing
questions — is unresolved and stays listed in Known Limitations below; for ongoing
personal use, new documents will keep arriving and this exact scenario will recur.

## Known Limitations

This section stays honest and unsoftened on purpose — it's a more useful signal of
where the system actually stands than a polished feature list would be.

- **Citation visibility gap.** The LLM sees and may cite up to 5 excerpts (`CONTEXT_K`),
  but only the top 3 (`CITATION_K`) are returned as citation cards. An answer can contain
  an inline `[4]` or `[5]` reference with no corresponding card shown to the user.
- **Duplicate filenames aren't disambiguated in citations.** The current index has two
  different files both named `attention.pdf` — citations show `source_file` (basename)
  only, so a user can't tell which one an answer actually came from without opening the
  ingestion status view (which does track full paths internally).
- **The "Strict grounding" toggle in the UI is decorative.** It's always shown on and
  disabled — the guardrail behavior it describes is real and always active, but the
  toggle itself doesn't control anything; there's no non-strict mode to switch to.
- **Feedback buttons don't persist anything.** They're wired into the UI and disclose
  this honestly on click, but there's no feedback endpoint or storage behind them yet.
- **No deleted-file pruning.** If a file is removed from the scan folder, its chunks
  stay in the index indefinitely — incremental ingestion only handles new and changed
  files, not removed ones. Confirmed directly on 2026-09-17: removing 15 files required
  a manual one-off pruning script rather than anything the running system did on its own.
- **No automated drift detection.** A newly-ingested document can legitimately out-rank
  an existing document for questions the eval set was calibrated against, silently
  changing the measured accuracy with no alert — this is exactly what happened on
  2026-09-17 (see the dated note under Measured Accuracy above). Nothing currently
  watches for this between manual `eval.py` runs.
- **The watcher scans its target folder recursively, including any subfolder you create
  inside it.** A folder meant to "exclude" files from the index must live outside
  `LOCAL_SCAN_DIR` entirely — a subfolder of it is still watched and will be re-ingested.
- **Untested at scale.** The system has been verified against ~100 documents / ~1,750
  chunks. Retrieval quality, FAISS index size, and re-embedding time at 10,000+ documents
  are unknown.
- **No automated regression suite.** `eval.py` checks retrieval/citation accuracy on
  demand; there's no CI, no unit tests, and no automated run on every change — a
  regression would only be caught by manually re-running the eval.
- **Role-based access is a folder-naming convention**, not real permissions (see above)
  — anyone with filesystem access to the scan folder controls what gets tagged
  finance-only.

## Sharing a live link (demo only)

For a pitch call where someone else needs to reach the running server:

```bash
ngrok http 8000
```

Share the `https://*.ngrok-free.app` URL it prints. This exposes the API and UI over
the internet for as long as ngrok runs — stop it when the demo is done. Not intended
for the always-on personal-use setup.

## What's deliberately out of scope

| Not built | Why |
|---|---|
| Email/IMAP ingestion | Built early, then removed entirely at the user's direction — no code remains |
| Real authentication / production RBAC | Role stub is folder-name-based, demo-only |
| Cloud LLM/embedding fallback | Hard requirement: fully local via Ollama |
| Multi-user concurrency handling | Single-user local tool |
| Deleted-file cleanup in the index | See Known Limitations |

## Honest performance note

Response time depends heavily on which LLM is configured and available RAM headroom.
`llama3.2:3b` on the development machine (8GB RAM) answers in the single-to-low-double
digits of seconds. `llama3.1:8b` on the same machine caused swap thrashing and 100–320s
responses — it's usable, but only recommended on a machine with more RAM to spare.
