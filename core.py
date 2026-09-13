"""
LangChain-based engine. LangChain's own Embeddings / VectorStore / LLM base
classes (OllamaEmbeddings, FAISS, OllamaLLM) already ARE the swappable-provider
abstraction this project needs, so there is no separate hand-rolled ABC layer
on top of them -- that would just be indirection wrapping indirection.

Swapping a provider later means changing the two get_*() functions below (or
pointing FAISS at a different LangChain vectorstore integration) -- nothing
in ingest.py / retrieval.py / app.py has to change.
"""
import json
import os
import re
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM

load_dotenv()

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "llama3.2:3b")

INDEX_DIR = Path(__file__).parent / "index"
INDEX_DIR.mkdir(exist_ok=True)
FAISS_DIR = str(INDEX_DIR / "faiss_store")
CHUNKS_PATH = INDEX_DIR / "chunks.json"

ROLES = ["all", "finance"]


def assign_role(file_path: str) -> str:
    """Demo-only RBAC stand-in: a file under a folder literally named
    'finance' (case-insensitive) is tagged Finance-only; everything else is
    visible to All Staff. Real RBAC would come from the source system's own
    permissions, not a folder-name convention."""
    parts = {p.lower() for p in Path(file_path).parts}
    return "finance" if "finance" in parts else "all"


def tokenize(text: str) -> List[str]:
    """Lowercase, punctuation-stripped tokens. BM25Retriever's own default
    preprocess_func is a plain str.split(), which would treat 'AutoMind?' and
    'automind' as different tokens -- passed explicitly as preprocess_func in
    retrieval.py to avoid that regression (bug fix #2)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)


def get_llm() -> OllamaLLM:
    return OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_HOST,
        # Bug fix #4: a thorough, multi-excerpt answer needs real generation
        # headroom -- Ollama's unset default is small enough to clip one.
        num_predict=700,
        # Bug fix #4 follow-on: a 700-token answer needs more request runway
        # than a short-answer default timeout gives it, especially loaded.
        sync_client_kwargs={"timeout": 600},
    )


def save_chunks(docs: List[Document]) -> None:
    with open(CHUNKS_PATH, "w") as f:
        json.dump([{"page_content": d.page_content, "metadata": d.metadata} for d in docs], f)


def load_chunks() -> List[Document]:
    if not CHUNKS_PATH.exists():
        return []
    with open(CHUNKS_PATH) as f:
        return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in json.load(f)]
