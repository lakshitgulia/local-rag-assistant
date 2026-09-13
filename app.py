"""
Query API + static chat UI. Loads the persisted FAISS index + chunk metadata
built by ingest.py once at startup, rebuilds the BM25 retriever from that same
metadata (cheap, CPU-only, no network call), then answers each /query request
with LangChain hybrid retrieval + a context-constrained Ollama generation.

Run: uvicorn app:app --host 0.0.0.0 --port 8000
"""
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel

from core import FAISS_DIR, ROLES, get_embeddings, get_llm, load_chunks
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
