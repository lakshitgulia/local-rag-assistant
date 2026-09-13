"""
Hybrid retrieval built on LangChain's EnsembleRetriever (BM25Retriever +
a FAISS vector retriever) followed by a context-constrained LLM answer
chained with LCEL (prompt | llm).

EnsembleRetriever was verified (by reading its source in
langchain_classic/retrievers/ensemble.py, not assumed) to implement the same
algorithm the hand-written fusion used: weighted Reciprocal Rank Fusion with
the same c=60 constant. So it replaces the hand-written RRF outright rather
than wrapping it.

The guardrail (decline when nothing relevant is found) and the top-3 citation
formatting are project-specific -- LangChain has no built-in for either, so
they stay as plain custom code operating on the retriever's output, per the
project's own design brief. That's also why this isn't a single unbroken
`retriever | prompt | llm` pipe: the guardrail needs to inspect the retrieved
documents *before* deciding whether to call the LLM at all, and the citation
builder needs the same documents *after* generation. Retrieval+filtering runs
as a normal function call; only the prompt->generate step is an LCEL chain.
"""
from dataclasses import dataclass
from typing import List

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from core import tokenize

VECTOR_K = 20
BM25_K = 20
CONTEXT_K = 5  # chunks handed to the LLM
CITATION_K = 3  # sources shown to the user
RRF_K = 60  # standard reciprocal-rank-fusion constant (matches EnsembleRetriever's default c)
NO_ANSWER = "I don't have information on that in the indexed documents."
# Small local models tend to paraphrase the exact fallback sentence (e.g. adding
# the topic into it), so grounding is detected by this looser phrase instead of
# an exact-string match -- see answer_question(). Bug fix #3.
NO_ANSWER_MARKER = "don't have information"

PROMPT = PromptTemplate.from_template(
    """You are a document assistant. Answer the question using ONLY the numbered excerpts below. \
Do not use any outside knowledge, and do not use anything you know beyond these excerpts, even if you \
recognize the topic from general knowledge. \
If the excerpts do not contain the answer, respond with ONLY this exact sentence and nothing else: \
"{no_answer}"

Each numbered excerpt may come from a DIFFERENT source document -- they are not all about the same \
topic just because they appear together. Before using an excerpt, check whether it actually supports \
what you're about to say. Do not describe an excerpt as part of a document it did not come from, and \
do not blend unrelated excerpts together as if they were one source.

Answer thoroughly using the specific details, numbers, and facts present in the sources below. \
Do not give a one-line summary -- explain the relevant details and reasoning a staff member would \
actually need, drawing on every excerpt that's relevant, not just the first one.

When you use information from an excerpt, cite it inline as [1], [2], etc. matching its number below.

{context}

Question: {question}

Answer:"""
)


@dataclass
class Source:
    rank: int
    file: str
    location: str
    snippet: str


@dataclass
class QueryResult:
    answer: str
    sources: List[Source]
    grounded: bool


def build_ensemble_retriever(vectorstore: FAISS, chunks: List[Document]) -> EnsembleRetriever:
    bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=tokenize)
    bm25_retriever.k = BM25_K
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": VECTOR_K})
    return EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5], c=RRF_K)


def _visible(doc: Document, role: str) -> bool:
    """Demo role-stub filter: 'all' role only sees non-restricted chunks,
    'finance' role sees everything. Not real access control."""
    doc_role = doc.metadata.get("role", "all")
    return doc_role == "all" or doc_role == role


def hybrid_search(question: str, role: str, retriever: EnsembleRetriever) -> List[Document]:
    fused = retriever.invoke(question)
    visible = [d for d in fused if _visible(d, role)]
    return visible[:CONTEXT_K]


def _format_context(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{i + 1}] (from {d.metadata['source_file']}, {d.metadata['location']})\n{d.page_content}"
        for i, d in enumerate(docs)
    )


def answer_question(question: str, role: str, retriever: EnsembleRetriever, llm) -> QueryResult:
    context_docs = hybrid_search(question, role, retriever)

    if not context_docs:
        return QueryResult(answer=NO_ANSWER, sources=[], grounded=False)

    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "no_answer": NO_ANSWER,
        "context": _format_context(context_docs),
        "question": question,
    })

    grounded = NO_ANSWER_MARKER not in answer.lower()
    sources = [
        Source(
            rank=i + 1,
            file=d.metadata["source_file"],
            location=d.metadata["location"],
            snippet=d.page_content[:200].strip() + "...",
        )
        for i, d in enumerate(context_docs[:CITATION_K])
    ] if grounded else []

    return QueryResult(answer=answer, sources=sources, grounded=grounded)
