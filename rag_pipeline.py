"""
RAG Pipeline - complete script.

Two modes:
  python rag_pipeline.py index <folder_of_pdfs>     -> builds/updates the vector store
  python rag_pipeline.py ask "your question here"   -> retrieves + answers

See README.md for setup instructions.
"""

import sys
import os
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------------------------------------------------------------------
# Config -- tune these as you learn how your documents behave
# ---------------------------------------------------------------------------
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_chunks"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE_WORDS = 500
CHUNK_OVERLAP_WORDS = 50
TOP_K = int(os.environ.get("TOP_K", 5))
MAX_DISTANCE = float(os.environ.get("MAX_DISTANCE", 0.25))  # cosine distance cutoff; chunks less similar than this are dropped as noise, even if top_k hasn't been filled. Lower = stricter. Tuned empirically via eval_threshold.py against the 20-PDF test set -- 0.20-0.25 was perfectly clean (20/20, zero noise), 0.30+ started leaking irrelevant sources. Re-run eval_threshold.py against your real documents once indexed; this exact number is specific to the synthetic test corpus and likely needs recalibrating for your actual PDFs. Override via .env (TOP_K, MAX_DISTANCE) or per-request in /ask.
HASH_CACHE_FILE = "./chroma_db/indexed_files.json"  # kept alongside the vector store so one volume/folder persists both

# Which LLM answers the question, once relevant chunks are retrieved.
# Set via .env: LLM_PROVIDER=anthropic (default) or LLM_PROVIDER=gemini
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").lower()
LLM_MODEL_OVERRIDE = os.environ.get("LLM_MODEL")  # optional, skips the default below
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-3.1-flash-lite",
}

# ---------------------------------------------------------------------------
# Step 1: Parsing
# ---------------------------------------------------------------------------

def extract_text_by_page(pdf_path: str) -> list[dict]:
    """Returns [{'page_num': int, 'text': str}, ...] for one PDF.
    Pages that fail to extract (corrupt page, etc.) are skipped with a warning
    rather than crashing the whole batch."""
    pages = []
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        print(f"  [WARN] Could not open {pdf_path}: {e}")
        return pages

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"  [WARN] Could not extract page {i+1} of {pdf_path}: {e}")
            text = ""
        pages.append({"page_num": i + 1, "text": text})
    return pages


def looks_like_scanned(pages: list[dict], min_chars_per_page: int = 20) -> bool:
    """Heuristic: if extracted text is suspiciously short, the PDF is
    probably a scanned image with no real text layer. Flag it instead
    of silently indexing near-empty chunks."""
    if not pages:
        return True
    avg_chars = sum(len(p["text"]) for p in pages) / len(pages)
    return avg_chars < min_chars_per_page


# ---------------------------------------------------------------------------
# Step 2: Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS,
               overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Word-count chunker with overlap so a sentence split across a
    chunk boundary isn't lost entirely from either chunk."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# Step 3: Embedding + storage
# ---------------------------------------------------------------------------

_model = None
_client = None
_collection = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model ({EMBEDDING_MODEL})... first run downloads it.")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        # Explicit cosine space: distance = 1 - similarity, so 0 = identical
        # meaning, 1 = unrelated. Without this, Chroma defaults to L2 distance,
        # which works fine for ranking but isn't as intuitive to set a
        # noise-filtering threshold against.
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def file_hash(path: Path) -> str:
    """MD5 of file contents -- used to detect changed files for incremental indexing."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_hash_cache() -> dict:
    if Path(HASH_CACHE_FILE).exists():
        return json.loads(Path(HASH_CACHE_FILE).read_text())
    return {}

def save_hash_cache(cache: dict):
    Path(HASH_CACHE_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(HASH_CACHE_FILE).write_text(json.dumps(cache, indent=2))


def embed_and_store(records: list[dict], batch_size: int = 64):
    """records: [{'text', 'source', 'page'}, ...]"""
    model = get_model()
    collection = get_collection()

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        texts = [r["text"] for r in batch]
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()

        collection.add(
            ids=[f"{Path(r['source']).stem}_p{r['page']}_c{i+j}" for j, r in enumerate(batch)],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": r["source"], "page": r["page"]} for r in batch],
        )


def index_pdf(pdf_path: Path) -> int:
    """Indexes one PDF. Returns number of chunks added (0 if skipped/scanned)."""
    pages = extract_text_by_page(str(pdf_path))

    if looks_like_scanned(pages):
        print(f"  [SKIP] {pdf_path.name} looks like a scanned PDF (little/no text). "
              f"Needs OCR -- see README for how to add that.")
        return 0

    records = []
    for p in pages:
        for chunk in chunk_text(p["text"]):
            records.append({"text": chunk, "source": str(pdf_path), "page": p["page_num"]})

    if records:
        embed_and_store(records)
    return len(records)


def index_folder(folder: str):
    folder_path = Path(folder)
    pdf_files = sorted(folder_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {folder}")
        return

    cache = load_hash_cache()
    total_chunks = 0
    skipped_unchanged = 0

    for pdf_path in pdf_files:
        h = file_hash(pdf_path)
        key = str(pdf_path)
        if cache.get(key) == h:
            skipped_unchanged += 1
            continue  # unchanged since last run -- don't re-embed

        print(f"Indexing {pdf_path.name}...")
        n_chunks = index_pdf(pdf_path)
        total_chunks += n_chunks
        cache[key] = h  # mark as indexed at this hash

    save_hash_cache(cache)
    print(f"\nDone. {total_chunks} new chunks indexed. "
          f"{skipped_unchanged} files unchanged and skipped.")


# ---------------------------------------------------------------------------
# Step 4 + 5: Retrieval + generation
# ---------------------------------------------------------------------------

def retrieve(question: str, top_k: int = TOP_K, max_distance: float = MAX_DISTANCE) -> list[dict]:
    model = get_model()
    collection = get_collection()

    query_embedding = model.encode([question], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"] or not results["documents"][0]:
        return []

    chunks = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        if dist > max_distance:
            continue  # too dissimilar to the question -- treat as noise, not a real match
        chunks.append({"text": doc, "source": meta["source"], "page": meta["page"], "distance": dist})
    return chunks


def generate_answer(prompt: str) -> str:
    """Sends the prompt to whichever LLM is configured via LLM_PROVIDER.
    Both providers get the exact same prompt -- only the API call differs,
    so switching providers never changes retrieval quality, only generation."""
    key = os.environ.get("LLM_KEY") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "No LLM API key found. Set LLM_KEY (and LLM_PROVIDER) in your .env file."
        )

    model = LLM_MODEL_OVERRIDE or DEFAULT_MODELS.get(LLM_PROVIDER)
    if model is None:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'anthropic' or 'gemini'."
        )

    if LLM_PROVIDER == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    elif LLM_PROVIDER == "gemini":
        from google import genai
        client = genai.Client(api_key=key)
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'anthropic' or 'gemini'."
        )


def answer_question(question: str, top_k: int = TOP_K, max_distance: float = MAX_DISTANCE) -> str:
    chunks = retrieve(question, top_k, max_distance)
    if not chunks:
        collection = get_collection()
        if collection.count() == 0:
            return "No indexed content found. Did you run 'index' first?"
        return ("No sufficiently relevant content found for this question "
                "(everything retrieved was below the similarity threshold). "
                "Try rephrasing, or lower MAX_DISTANCE if this keeps happening on valid questions.")

    context = "\n\n".join(
        f"[Source: {Path(c['source']).name}, page {c['page']}]\n{c['text']}"
        for c in chunks
    )

    prompt = f"""Answer the question using ONLY the context below.
If the context doesn't contain the answer, say so clearly -- don't guess.
Do NOT include filenames, page numbers, or source citations in your answer --
just answer the question directly in plain language. Source attribution is
handled separately and shown alongside your answer.

Context:
{context}

Question: {question}"""

    return generate_answer(prompt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "index":
        if len(sys.argv) < 3:
            print("Usage: python rag_pipeline.py index <folder_of_pdfs>")
            sys.exit(1)
        index_folder(sys.argv[2])

    elif command == "ask":
        if len(sys.argv) < 3:
            print('Usage: python rag_pipeline.py ask "your question"')
            sys.exit(1)
        question = sys.argv[2]
        print(f"\nQ: {question}\n")
        print(answer_question(question))

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
