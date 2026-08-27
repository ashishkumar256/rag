#!/usr/bin/env python3
"""
Ingest Database Service Provider PDFs into ChromaDB using LangChain.

Usage:
  python ingest.py                  # ingest all PDFs
  python ingest.py --force          # wipe collection and re-ingest
  python ingest.py --query "SLA encryption" --k 3
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config (overridable via env)
# ---------------------------------------------------------------------------
PDF_DIR = Path(os.getenv("PDF_DIR", "/data/pdfs"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "/data/chroma"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "db_providers")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("ingest")


class LocalEmbeddings(Embeddings):
    """Thin LangChain-compatible wrapper around sentence-transformers."""

    def __init__(self, model_name: str = EMBED_MODEL):
        log.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        log.info("Embedding model ready")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def load_and_chunk(pdf_path: Path, splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    """Load one PDF with LangChain PyPDFLoader and split into chunks."""
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()  # one Document per page, metadata has 'page' and 'source'

    # Enrich metadata
    stem = pdf_path.stem
    company_slug = stem.split("_", 1)[1] if "_" in stem else stem
    fp = file_hash(pdf_path)

    for doc in pages:
        doc.metadata["source_file"] = pdf_path.name
        doc.metadata["company_slug"] = company_slug
        doc.metadata["file_hash"] = fp
        # page is already set by PyPDFLoader (0-based in some versions; keep as-is)

    chunks = splitter.split_documents(pages)
    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
        c.metadata["chunk_id"] = f"{stem}__{fp}__{i}"
    return chunks


def ingest(force: bool = False) -> None:
    if not PDF_DIR.exists():
        log.error("PDF_DIR does not exist: %s", PDF_DIR)
        sys.exit(1)

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        log.error("No PDF files found in %s", PDF_DIR)
        sys.exit(1)

    log.info("Found %d PDF(s) in %s", len(pdfs), PDF_DIR)

    embeddings = LocalEmbeddings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    if force and (CHROMA_DIR / "chroma.sqlite3").exists():
        log.warning("Force=True → deleting existing Chroma data at %s", CHROMA_DIR)
        import shutil
        shutil.rmtree(CHROMA_DIR)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # Optional: skip already-ingested files by checking existing metadata
    existing_files = set()
    if not force:
        try:
            # cheap way: get a few docs and collect source_file values
            sample = vectorstore.get(include=["metadatas"], limit=10000)
            for m in sample.get("metadatas") or []:
                if m and "source_file" in m:
                    existing_files.add(m["source_file"])
            if existing_files:
                log.info("Already ingested files: %s", sorted(existing_files))
        except Exception:
            pass

    total_chunks = 0
    for pdf in pdfs:
        if pdf.name in existing_files and not force:
            log.info("Skipping already-ingested %s", pdf.name)
            continue

        log.info("Parsing & chunking %s ...", pdf.name)
        chunks = load_and_chunk(pdf, splitter)
        if not chunks:
            log.warning("No chunks produced for %s", pdf.name)
            continue

        log.info("  → %d chunks, embedding & storing ...", len(chunks))
        vectorstore.add_documents(chunks)
        total_chunks += len(chunks)

    count = vectorstore._collection.count()
    log.info("Done. Collection '%s' now holds %d vectors (this run added ~%d chunks).",
             COLLECTION_NAME, count, total_chunks)


def search(query: str, k: int = 5) -> None:
    embeddings = LocalEmbeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    results = vectorstore.similarity_search_with_score(query, k=k)
    print(f"\nQuery: {query!r}\n")
    for rank, (doc, score) in enumerate(results, 1):
        meta = doc.metadata
        print(f"--- Rank {rank}  (distance={score:.4f}) ---")
        print(f"  source : {meta.get('source_file')}  page={meta.get('page')}  company={meta.get('company_slug')}")
        print(f"  text   : {doc.page_content[:400].replace(chr(10), ' ')}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB (LangChain)")
    parser.add_argument("--force", action="store_true", help="Wipe and re-ingest everything")
    parser.add_argument("--query", type=str, help="Run a similarity search instead of ingest")
    parser.add_argument("--k", type=int, default=5, help="Top-k results for --query")
    args = parser.parse_args()

    if args.query:
        search(args.query, k=args.k)
    else:
        ingest(force=args.force)


if __name__ == "__main__":
    main()
