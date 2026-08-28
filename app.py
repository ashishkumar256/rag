#!/usr/bin/env python3
"""
RAG service:
  - Connects to remote ChromaDB container
  - LangChain: parse PDFs → chunk → embed (fastembed) → store
  - Exposes /health, /ingest, /search, /documents
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import chromadb
import uvicorn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PDF_DIR = Path(os.getenv("PDF_DIR", "/data/pdfs"))
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "db_providers")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rag")

app = FastAPI(title="DB Providers RAG", version="2.0.0")

_embedder: Optional["FastEmbedEmbeddings"] = None
_vs: Optional[Chroma] = None
ingest_status: Dict[str, Any] = {"status": "idle", "message": "", "docs_processed": 0, "chunks": 0}


class FastEmbedEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBED_MODEL):
        from fastembed import TextEmbedding
        log.info("Loading fastembed model: %s", model_name)
        self.model = TextEmbedding(model_name=model_name)
        log.info("Embedding model ready")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [e.tolist() for e in self.model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return list(self.model.embed([text]))[0].tolist()


def get_chroma_client() -> chromadb.HttpClient:
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


def get_vectorstore() -> Chroma:
    global _vs, _embedder
    if _vs is None:
        _embedder = FastEmbedEmbeddings()
        client = get_chroma_client()
        _vs = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=_embedder,
        )
    return _vs


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def load_and_chunk(pdf_path: Path, splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    stem = pdf_path.stem
    company_slug = stem.split("_", 1)[1] if "_" in stem else stem
    fp = file_hash(pdf_path)

    for doc in pages:
        doc.metadata["source_file"] = pdf_path.name
        doc.metadata["company_slug"] = company_slug
        doc.metadata["file_hash"] = fp

    chunks = splitter.split_documents(pages)
    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
        c.metadata["chunk_id"] = f"{stem}__{fp}__{i}"
    return chunks


def run_ingest(force: bool = False) -> Dict[str, Any]:
    global ingest_status
    ingest_status = {"status": "running", "message": "Starting...", "docs_processed": 0, "chunks": 0}

    if not PDF_DIR.exists():
        ingest_status = {"status": "error", "message": f"PDF_DIR missing: {PDF_DIR}", "docs_processed": 0, "chunks": 0}
        return ingest_status

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        ingest_status = {"status": "error", "message": "No PDFs found", "docs_processed": 0, "chunks": 0}
        return ingest_status

    vs = get_vectorstore()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    if force:
        try:
            client = get_chroma_client()
            client.delete_collection(COLLECTION_NAME)
            log.info("Deleted collection %s", COLLECTION_NAME)
            # recreate
            global _vs
            _vs = None
            vs = get_vectorstore()
        except Exception as e:
            log.warning("Could not delete collection: %s", e)

    existing = set()
    if not force:
        try:
            sample = vs.get(include=["metadatas"], limit=20000)
            for m in sample.get("metadatas") or []:
                if m and "source_file" in m:
                    existing.add(m["source_file"])
        except Exception:
            pass

    total_chunks = 0
    docs_ok = 0
    for pdf in pdfs:
        if pdf.name in existing and not force:
            log.info("Skip already-ingested %s", pdf.name)
            continue
        log.info("Processing %s", pdf.name)
        chunks = load_and_chunk(pdf, splitter)
        if not chunks:
            continue
        vs.add_documents(chunks)
        total_chunks += len(chunks)
        docs_ok += 1

    ingest_status = {
        "status": "done",
        "message": f"Ingested {docs_ok} files, {total_chunks} chunks",
        "docs_processed": docs_ok,
        "chunks": total_chunks,
    }
    log.info(ingest_status["message"])
    return ingest_status


# ---------- API models ----------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=30)


class IngestRequest(BaseModel):
    force: bool = False


# ---------- Routes ----------
@app.on_event("startup")
def startup():
    # Auto-ingest if collection is empty
    try:
        vs = get_vectorstore()
        count = vs._collection.count()
        log.info("Chroma collection has %d vectors", count)
        if count == 0:
            log.info("Empty collection → starting auto-ingest")
            run_ingest(force=False)
    except Exception as e:
        log.warning("Startup ingest check failed: %s", e)


@app.get("/health")
def health():
    try:
        client = get_chroma_client()
        hb = client.heartbeat()
        vs = get_vectorstore()
        count = vs._collection.count()
        pdf_count = len(list(PDF_DIR.glob("*.pdf"))) if PDF_DIR.exists() else 0
        return {
            "status": "ok",
            "chroma_host": f"{CHROMA_HOST}:{CHROMA_PORT}",
            "chroma_heartbeat": hb,
            "collection": COLLECTION_NAME,
            "vectors": count,
            "pdf_dir": str(PDF_DIR),
            "pdf_count": pdf_count,
            "embed_model": EMBED_MODEL,
            "ingest_status": ingest_status,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/ingest")
def ingest(body: IngestRequest = IngestRequest(), background: BackgroundTasks = None):
    result = run_ingest(force=body.force)
    return result


@app.get("/search")
def search_get(q: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=30)):
    return _search(q, k)


@app.post("/search")
def search_post(body: SearchRequest):
    return _search(body.query, body.k)


def _search(query: str, k: int):
    vs = get_vectorstore()
    results = vs.similarity_search_with_score(query, k=k)
    hits = []
    for rank, (doc, dist) in enumerate(results, 1):
        meta = doc.metadata or {}
        hits.append({
            "rank": rank,
            "score": float(1.0 - dist) if dist <= 1 else float(dist),
            "text": doc.page_content,
            "source_file": meta.get("source_file"),
            "company_slug": meta.get("company_slug"),
            "page": meta.get("page"),
            "chunk_index": meta.get("chunk_index"),
        })
    return {"query": query, "k": k, "results": hits}


@app.get("/documents")
def documents():
    vs = get_vectorstore()
    sample = vs.get(include=["metadatas"], limit=20000)
    files = {}
    for m in sample.get("metadatas") or []:
        if not m:
            continue
        src = m.get("source_file")
        if src:
            files.setdefault(src, {"source_file": src, "company_slug": m.get("company_slug"), "chunks": 0})
            files[src]["chunks"] += 1
    return {"documents": list(files.values()), "total_files": len(files)}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
