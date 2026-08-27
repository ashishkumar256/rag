#!/usr/bin/env python3
"""
Minimal FastAPI search service over the Chroma collection created by ingest.py.
"""

from __future__ import annotations

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
import uvicorn

CHROMA_DIR = os.getenv("CHROMA_DIR", "/data/chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "db_providers")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

app = FastAPI(title="DB Providers RAG Search", version="1.0.0")

_embedder: Optional[SentenceTransformer] = None
_vs: Optional[Chroma] = None


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBED_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()


def get_vs() -> Chroma:
    global _vs, _embedder
    if _vs is None:
        _embedder = LocalEmbeddings()
        _vs = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=_embedder,
            persist_directory=CHROMA_DIR,
        )
    return _vs


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=30)


class SearchHit(BaseModel):
    rank: int
    score: float
    text: str
    source_file: Optional[str] = None
    company_slug: Optional[str] = None
    page: Optional[int] = None
    chunk_index: Optional[int] = None


@app.get("/health")
def health():
    try:
        vs = get_vs()
        count = vs._collection.count()
        return {
            "status": "ok",
            "collection": COLLECTION_NAME,
            "vectors": count,
            "embed_model": EMBED_MODEL,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/search")
def search_get(q: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=30)):
    return _do_search(q, k)


@app.post("/search")
def search_post(body: SearchRequest):
    return _do_search(body.query, body.k)


def _do_search(query: str, k: int):
    vs = get_vs()
    results = vs.similarity_search_with_score(query, k=k)
    hits = []
    for rank, (doc, dist) in enumerate(results, 1):
        meta = doc.metadata or {}
        hits.append(
            SearchHit(
                rank=rank,
                score=float(1.0 - dist) if dist <= 1 else float(dist),  # rough similarity
                text=doc.page_content,
                source_file=meta.get("source_file"),
                company_slug=meta.get("company_slug"),
                page=meta.get("page"),
                chunk_index=meta.get("chunk_index"),
            )
        )
    return {"query": query, "k": k, "results": hits}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
