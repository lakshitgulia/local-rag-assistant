"""
Query API + static chat UI. Loads the persisted FAISS index + chunk metadata
built by ingest.py once at startup, rebuilds the BM25 retriever from that same
metadata (cheap, CPU-only, no network call), then answers each /query request
with LangChain hybrid retrieval + a context-constrained Ollama generation.

Run: uvicorn app:app --host 0.0.0.0 --port 8000
"""
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel

from core import FAISS_DIR, MANIFEST_PATH, ROLES, get_embeddings, get_llm, load_chunks, load_manifest, load_watcher_status
from retrieval import answer_question, build_ensemble_retriever

app = FastAPI(title="RAG Assistant Demo")
app.mount("/static", StaticFiles(directory="static"), name="static")

llm = get_llm()
chunks = load_chunks()
retriever = None

if chunks:
    vectorstore = FAISS.load_local(FAISS_DIR, get_embeddings(), allow_dangerous_deserialization=True)
    retriever = build_ensemble_retriever(vectorstore, chunks)


class QueryRequest(BaseModel):
    question: str
    role: str = "all"


class SourceResponse(BaseModel):
    rank: int
    file: str
    location: str
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    grounded: bool
    response_time_seconds: float


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/roles")
def roles():
    return {"roles": ROLES}


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not chunks or retriever is None:
        raise HTTPException(status_code=503, detail="No documents indexed yet. Run `python ingest.py` first.")
    if request.role not in ROLES:
        raise HTTPException(status_code=400, detail=f"role must be one of {ROLES}")

    start = time.time()
    result = answer_question(request.question, request.role, retriever, llm)
    elapsed = time.time() - start
    print(f"[app] query={request.question!r} role={request.role} elapsed={elapsed:.2f}s grounded={result.grounded}")

    return QueryResponse(
        answer=result.answer,
        sources=[SourceResponse(**s.__dict__) for s in result.sources],
        grounded=result.grounded,
        response_time_seconds=round(elapsed, 2),
    )


class StatsResponse(BaseModel):
    total_documents: int
    total_chunks: int


@app.get("/api/stats", response_model=StatsResponse)
def stats():
    """Real counts for the chat header's stat pill. Reads chunks.json fresh
    from disk rather than the in-memory `chunks` loaded at server startup --
    the watcher runs as a separate process and updates the index on disk
    without this server process reloading it, so the in-memory copy goes
    stale the moment the watcher indexes anything."""
    return StatsResponse(total_documents=len(load_manifest()), total_chunks=len(load_chunks()))


class WatcherUpdate(BaseModel):
    last_run_at: str
    last_file: str


class IngestStatusResponse(BaseModel):
    total_documents: int
    total_chunks: int
    files_by_type: dict[str, int]
    index_last_updated: Optional[str]
    last_watcher_update: Optional[WatcherUpdate]


@app.get("/api/ingest/status", response_model=IngestStatusResponse)
def ingest_status():
    """Real ingestion state for the Document Ingestion view -- reads
    index/manifest.json (built by the incremental-ingestion work) and
    index/watcher_status.json (written by watcher.py). last_watcher_update is
    None until the watcher has actually processed a file -- never fabricated."""
    manifest = load_manifest()

    files_by_type: dict[str, int] = {}
    for path_str in manifest:
        ext = Path(path_str).suffix.lower().lstrip(".") or "unknown"
        files_by_type[ext] = files_by_type.get(ext, 0) + 1

    index_last_updated = None
    if MANIFEST_PATH.exists():
        index_last_updated = datetime.fromtimestamp(MANIFEST_PATH.stat().st_mtime, tz=timezone.utc).isoformat()

    watcher_status = load_watcher_status()

    return IngestStatusResponse(
        total_documents=len(manifest),
        total_chunks=len(load_chunks()),  # fresh from disk -- see /api/stats
        files_by_type=files_by_type,
        index_last_updated=index_last_updated,
        last_watcher_update=WatcherUpdate(**watcher_status) if watcher_status else None,
    )
