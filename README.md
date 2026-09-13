# RAG Assistant Demo (Proof of Concept)

A scoped-down demo built to show a bid client the mechanics of a RAG assistant
over their internal documents: ingestion → hybrid retrieval → grounded answer
with citations. **100% local inference via Ollama — zero external network
calls for embeddings or generation.**

## Architecture

```
ingest.py     -- one-time batch pipeline: parse -> chunk -> embed -> index
core.py       -- get_embeddings()/get_llm() (LangChain wrappers around Ollama),
                 tokenize(), demo role rule, chunk persistence
retrieval.py  -- hybrid search via LangChain's EnsembleRetriever (BM25 +
                 FAISS, reciprocal rank fusion) and an LCEL prompt|llm chain
app.py        -- FastAPI query endpoint + serves the chat UI
eval.py       -- top-3 citation-accuracy eval harness
static/index.html  -- single-page chat UI
notebooks/    -- numbered demo walkthroughs (see "Notebooks" below)
```

`core.py` exposes `get_embeddings()` / `get_llm()`, which return LangChain's
own `OllamaEmbeddings` / `OllamaLLM`. LangChain's `Embeddings`/`VectorStore`/
`LLM` base classes are themselves the swappable-provider abstraction — there's
no separate hand-written interface layer on top of them. `retrieval.py` builds
a LangChain `EnsembleRetriever` (a `BM25Retriever` + a FAISS vector retriever,
combined by weighted reciprocal rank fusion, `c=60`) and chains the prompt and
LLM call with LCEL (`prompt | llm`). Swapping to a hosted model or a different
vector store later means changing `get_embeddings()`/`get_llm()` or pointing
at a different LangChain vectorstore integration — nothing in `ingest.py`,
`retrieval.py`, or `app.py` otherwise changes.

## Prerequisites

- [Ollama](https://ollama.com) installed and running
- Models pulled: `ollama pull nomic-embed-text` and `ollama pull llama3.2:3b`
  (check with `ollama list`). `llama3.2:3b` is the default — chosen for speed
  on modest hardware (8GB RAM here caused `llama3.1:8b` to swap-thrash, taking
  100-320s per query). `llama3.1:8b` is not required; if you have more RAM/GPU
  headroom, pull it separately and switch `OLLAMA_LLM_MODEL` in `.env`.
- Python 3.10+
- `tesseract` and `poppler` (for OCR of scanned-page images/PDFs):
  `brew install tesseract poppler`

## Getting your documents in

There's no manual `sample_data/` folder to populate — this demo pulls your
own real files. Set `LOCAL_SCAN_DIR` in `.env` (copy from `.env.example`) to
a real folder on your machine containing PDFs, PPTX, or DOCX files — e.g.
`~/Downloads`. Running `python ingest.py` scans it recursively. This is a
one-time, run-on-demand batch load — not a live sync.

Scanned-page images (`.png/.jpg/.jpeg/.tiff/.bmp`) and image-only PDF pages
are OCR'd automatically via Tesseract. If a page/image yields effectively no
text, it's logged as flagged and skipped rather than guessed at — that's the
signal it's a photo/visual asset, not a scanned document, which stays out of
scope for this demo.

## Run it

```bash
./run.sh
```

This pulls any missing Ollama models, creates a venv, installs dependencies,
runs ingestion if `index/` is empty, and starts the server at
`http://0.0.0.0:8000`.

Open `http://localhost:8000`, ask a question, and you'll get a grounded
answer plus up to 3 cited sources (file name + page/slide/section + snippet).
Response time is shown under the answer and logged server-side.

To re-index after adding more files, delete `index/` (or just re-run
`python ingest.py` — it rebuilds from scratch each time).

## Role stub (demo stand-in, not real RBAC)

The UI has an "All Staff" / "Finance" dropdown. During ingestion, any file
sitting under a folder literally named `finance` is tagged Finance-only;
everything else is visible to All Staff. Selecting "All Staff" filters those
citations out of the results. This is a placeholder to gesture at
role-based access — the real thing would read permissions from the source
system (SharePoint/Exchange ACLs, etc.), not a folder-name convention.

## How hybrid retrieval + citations work

For each question: embed it and run FAISS vector similarity search (via
LangChain's FAISS vectorstore), and separately run BM25 keyword search (via
LangChain's `BM25Retriever`, wrapping `rank_bm25`) over the same chunks.
Both retrievers are combined by LangChain's `EnsembleRetriever`, which merges
them via weighted reciprocal rank fusion (no score-scale tuning needed). The
top 5 fused, role-visible chunks go to the LLM as the *only* allowed context
(via an LCEL `prompt | llm` chain); the top 3 are shown as citations. If
nothing relevant is retrieved, the model is instructed to say so explicitly
rather than answer from general knowledge — this is the guardrail, and it's
enforced by prompt instruction plus a loose substring check on the response,
not a separate filter, since the retrieval step already returns nothing to
answer from. The guardrail and citation formatting are custom code — LangChain
has no built-in for either.

## Built with LangChain

The retrieval/generation engine is built on LangChain rather than hand-rolled
HTTP calls:

- `langchain` / `langchain-core` — `Document`, `PromptTemplate`, LCEL chaining
- `langchain-community` — the `FAISS` vectorstore wrapper and `BM25Retriever`
- `langchain-ollama` — `OllamaEmbeddings` and `OllamaLLM`
- `langchain-classic` — `EnsembleRetriever` (moved here in LangChain 1.x,
  which reoriented the main `langchain` package around agents)

`langsmith` is installed as an **unavoidable transitive dependency** of
`langchain-core` (every version back to 0.3.0 requires it) but stays
completely inert here: no `LANGCHAIN_API_KEY`, no `LANGCHAIN_TRACING_V2`
anywhere in the code or environment. Verified, not just claimed — checking
live network connections during a query shows exactly two established TCP
connections, both `127.0.0.1:11434` (Ollama). Nothing else, anywhere.

## Notebooks

`notebooks/` holds 5 numbered walkthroughs of the pipeline, each calling the
real engine code against the real indexed corpus — no reimplemented logic:

1. `01_ingestion.ipynb` — real parsing on real files
2. `02_chunking.ipynb` — real chunking, shows actual chunk boundaries/sizes
3. `03_embedding_vectorstore.ipynb` — real embedding + FAISS similarity search
4. `04_retrieval.ipynb` — vector-only, BM25-only, and fused rankings side by side
5. `05_generation.ipynb` — full question → grounded answer → citations

These are a demo/companion layer only. The live app never imports anything
from `notebooks/`, and removing the folder entirely doesn't change how
`app.py` behaves.

## Measured accuracy

A hand-drafted set of 15 real questions against 9 distinct real files in the
indexed corpus (not resumes/hackathon decks) — 3 of them deliberately
adversarial (filename collisions, a chunk known to previously cause
cross-document content blending, a near-duplicate-filename check) — scored
**13/15 correct in the top-3 citations (87%)**, above the 80% target. All 3
adversarial questions passed. Run via `python eval.py eval_questions.json`.

The 2 misses, reported honestly rather than hidden:
- One source PDF has degraded text extraction (no spaces between words),
  which breaks keyword matching and likely hurts its embedding quality too —
  a source-data quality issue, not a retrieval-logic bug.
- One question's answer lived in a different chunk of the right document than
  the one retrieved; the model correctly declined rather than guess. DOCX
  files are chunked without page numbers, so this is the DOCX-equivalent of
  a "right document, wrong page" miss.

## Sharing a live link during the pitch (ngrok)

```bash
ngrok http 8000
```

Share the printed `https://*.ngrok-free.app` URL for the demo window. This
is a temporary tunnel to your local machine, not a cloud deployment — say so
explicitly if asked. The app already binds to `0.0.0.0:8000` so ngrok can
reach it.

## What's deliberately out of scope for this demo (and the plan to add it)

| Skipped here | Production approach |
|---|---|
| Live ingestion connectors | SharePoint Graph API / Exchange sync with webhooks or polling |
| Real SSO/RBAC | Real identity provider integration; permissions read from source ACLs |
| Cross-encoder reranker | Add a reranking pass on top of the existing hybrid retrieval |
| Slack/Teams bot | Bot Framework / Slack Bolt app calling the same query API |
| Chat history / feedback | Persistent store (Postgres) for sessions + thumbs up/down |
| Tool-agent / external lookups | Not planned unless the client's use case needs live data |
| Cloud deployment | Containerize and deploy to the client's VPC, GPU-backed inference |
| Formal load test / accuracy benchmark | Structured eval set + latency testing at ~10k docs |

## Honest performance note

The default model, `llama3.2:3b`, answers in roughly 7-20 seconds on this
hardware (CPU/modest GPU) — slower than a hosted API, but usable live.
`llama3.1:8b` is available as a larger, slower alternative (swap `.env`'s
`OLLAMA_LLM_MODEL`) if you're running on a machine with more RAM/GPU headroom
than the 8GB used here, where it caused swap-thrashing and 100-320s answers.
Either way this is a hardware tradeoff for the demo, not a design flaw:
vector search itself stays sub-100ms even at ~10k documents, and production
would run this on proper self-hosted GPU infrastructure with streaming to
keep answers feeling fast.
