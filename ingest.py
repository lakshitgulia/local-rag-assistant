"""
One-time batch ingestion: scan a local folder, parse PDF/PPTX/DOCX/EML/MBOX/
scanned-image files, chunk the text, embed it via Ollama, and persist a FAISS
index + chunk metadata under ./index/. (No separate BM25 index file:
retrieval.py rebuilds BM25Retriever fresh from chunks.json at app startup --
that's pure CPU, no network call, and cheap enough not to need its own
persistence format.)

Run on demand:
    python ingest.py                      # scans LOCAL_SCAN_DIR from .env
    python ingest.py --scan-dir ~/Desktop/demo-docs
"""
import argparse
import mailbox
import os
from email import message_from_bytes
from pathlib import Path
from typing import List, Tuple

import pytesseract
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pdf2image import convert_from_path
from PIL import Image
from pptx import Presentation
from pypdf import PdfReader
from docx import Document as DocxDocument

from core import FAISS_DIR, assign_role, get_embeddings, save_chunks

load_dotenv()

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
# Character-based (not word-based) so a PDF with broken text extraction --
# glyphs run together with no spaces -- can't produce one giant "word" that
# blows past Ollama's embedding context window.
CHUNK_CHARS = 2000  # ~400-500 tokens of normal English, ~10-15% overlap band
OVERLAP_RATIO = 0.13
MIN_OCR_CHARS = 20  # below this, a PDF page or image is treated as non-text and flagged


def chunk_text(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    step = int(CHUNK_CHARS * (1 - OVERLAP_RATIO))
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start:start + CHUNK_CHARS].strip()
        if chunk:
            chunks.append(chunk)
        if start + CHUNK_CHARS >= len(text):
            break
    return chunks


def parse_pdf(path: Path) -> List[Tuple[str, str]]:
    """Returns [(text, 'page N'), ...]. Falls back to OCR for pages with
    little/no extractable text (scanned pages)."""
    results = []
    reader = PdfReader(str(path))
    ocr_pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if len(text) < MIN_OCR_CHARS:
            ocr_pages.append(i)
        else:
            results.append((text, f"page {i + 1}"))

    if ocr_pages:
        try:
            images = convert_from_path(str(path))
            for i in ocr_pages:
                if i < len(images):
                    ocr_text = pytesseract.image_to_string(images[i]).strip()
                    if len(ocr_text) >= MIN_OCR_CHARS:
                        results.append((ocr_text, f"page {i + 1} (OCR)"))
                    else:
                        print(f"[ingest] flagged, no extractable text (visual/photo page?): {path.name} page {i + 1} -- skipped")
        except Exception as e:
            print(f"[ingest] OCR fallback failed for {path.name}: {e}")
    return results


def parse_pptx(path: Path) -> List[Tuple[str, str]]:
    results = []
    prs = Presentation(str(path))
    for i, slide in enumerate(prs.slides):
        texts = [shape.text for shape in slide.shapes if shape.has_text_frame and shape.text.strip()]
        if texts:
            results.append(("\n".join(texts), f"slide {i + 1}"))
    return results


def parse_docx(path: Path) -> List[Tuple[str, str]]:
    doc = DocxDocument(str(path))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [(text, "document")] if text.strip() else []


def parse_eml(path: Path) -> List[Tuple[str, str]]:
    with open(path, "rb") as f:
        msg = message_from_bytes(f.read())
    return _parse_email_message(msg)


def parse_mbox(path: Path) -> List[Tuple[str, str]]:
    results = []
    box = mailbox.mbox(str(path))
    for i, msg in enumerate(box):
        for text, _ in _parse_email_message(msg):
            results.append((text, f"message {i + 1}"))
    return results


def _parse_email_message(msg) -> List[Tuple[str, str]]:
    subject = msg.get("Subject", "(no subject)")
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and not part.get_filename():
                payload = part.get_payload(decode=True)
                if payload:
                    body += payload.decode(errors="ignore")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode(errors="ignore")
    text = f"Subject: {subject}\n\n{body}".strip()
    return [(text, "email")] if body.strip() else []


def parse_image(path: Path) -> List[Tuple[str, str]]:
    text = pytesseract.image_to_string(Image.open(path)).strip()
    if len(text) < MIN_OCR_CHARS:
        print(f"[ingest] flagged, no extractable text (visual/photo content?): {path.name} -- skipped, out of scope for this demo")
        return []
    return [(text, "image (OCR)")]


PARSERS = {
    ".pdf": parse_pdf,
    ".pptx": parse_pptx,
    ".docx": parse_docx,
    ".eml": parse_eml,
    ".mbox": parse_mbox,
}


def collect_files(scan_dir: Path) -> List[Path]:
    exts = set(PARSERS.keys()) | IMAGE_EXTS
    return [p for p in scan_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def build_chunks(files: List[Path]) -> List[Document]:
    all_chunks = []
    for path in files:
        ext = path.suffix.lower()
        parser = PARSERS.get(ext, parse_image if ext in IMAGE_EXTS else None)
        if parser is None:
            continue
        try:
            sections = parser(path)
        except Exception as e:
            print(f"[ingest] failed to parse {path.name}: {e}")
            continue

        role = assign_role(str(path))
        for text, location in sections:
            for piece in chunk_text(text):
                all_chunks.append(Document(
                    page_content=piece,
                    metadata={"source_file": path.name, "location": location, "role": role},
                ))
        print(f"[ingest] parsed {path.name} ({len(sections)} section(s))")
    return all_chunks


EMBED_BATCH_SIZE = 64  # LangChain's embed_documents() sends the whole list in
# one HTTP request; handing it all 882+ chunks at once crashed Ollama's
# /api/embed with a connection reset after ~30s. Batching client-side avoids
# that -- 64 matches what a production batch size would reasonably look like.


def run_ingestion(files: List[Path]):
    chunks = build_chunks(files)
    print(f"[ingest] built {len(chunks)} chunk(s) from {len(files)} file(s)")
    if not chunks:
        print("[ingest] no text extracted -- nothing to index.")
        return

    print("[ingest] embedding chunks via Ollama (nomic-embed-text)...")
    embeddings = get_embeddings()
    vectorstore = None
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i:i + EMBED_BATCH_SIZE]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        print(f"[ingest] embedded {min(i + EMBED_BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")
    vectorstore.save_local(FAISS_DIR)

    save_chunks(chunks)
    print(f"[ingest] done. Indexed {len(chunks)} chunks into ./index/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-dir", default=os.environ.get("LOCAL_SCAN_DIR", "~/Downloads"))
    parser.add_argument("--extensions", default=None,
                         help="Comma-separated list to restrict ingestion to (e.g. 'pdf,docx'). Default: all supported types.")
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir).expanduser()
    all_files = collect_files(scan_dir)

    if args.extensions:
        allowed = {f".{e.strip().lower().lstrip('.')}" for e in args.extensions.split(",")}
        all_files = [f for f in all_files if f.suffix.lower() in allowed]

    print(f"[ingest] found {len(all_files)} file(s) in {scan_dir}")

    if not all_files:
        print("[ingest] nothing to ingest -- add files to the scan folder and re-run.")
    else:
        run_ingestion(all_files)
