"""
app.py
------
Parses PDF vendor proposals, stores chunks in ChromaDB, and answers
questions against them. Every ChromaDB action (create collection, add,
count, query) is a real `curl` subprocess call — not the chromadb SDK.
`import chromadb` is present but unused for any DB call.

LLM_PROVIDER controls whether a generation step runs at all:

    LLM_PROVIDER=none   (default)  -> pure retrieval. The "answer" is
                                       the verbatim top-k chunks, exactly
                                       as before. No model call of any
                                       kind happens.

    LLM_PROVIDER=<anything else>   -> reserved for future wiring. This
                                       build does not implement any
                                       provider; setting one raises a
                                       clear error rather than silently
                                       falling back to retrieval-only,
                                       so you always know which mode
                                       actually ran.

Set it via environment variable or --llm-provider flag (flag wins).

No eval harness, no test-PDF generator, no answer key. Ingest PDFs
from a real folder yourself; query them; get chunks back.

Usage:
    python app.py ingest --pdf-dir pdfs
    python app.py query --q "What is the uptime SLA?"
    python app.py query --q "..." --top-k 5
    LLM_PROVIDER=none python app.py query --q "..."       # explicit, same as default
    python app.py --verbose ingest --pdf-dir pdfs          # print every curl command run
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import chromadb  # noqa: F401  -- imported as requested; not used for any DB call below
import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import HashingVectorizer

HOST = "localhost"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}/api/v2"
TENANT_DB = "tenants/default_tenant/databases/default_database"
COLLECTION_NAME = "db_vendor_proposals"
CHROMA_DATA_DIR = Path("chroma_data")
COLLECTION_ID_FILE = Path(".collection_id")

VERBOSE = False


# --------------------------------------------------------------------------
# curl wrapper — every DB action below is literally this function
# --------------------------------------------------------------------------

def curl(method: str, path: str, json_body: dict | None = None) -> dict:
    cmd = ["curl", "-s", "-X", method, f"{BASE_URL}/{path}"]
    if json_body is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(json_body)]

    if VERBOSE:
        printable = " ".join(f"'{c}'" if " " in c or "{" in c else c for c in cmd)
        print(f"$ {printable}", file=sys.stderr)

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if not result.stdout.strip():
        return {}
    return json.loads(result.stdout)


def curl_status_code(path: str) -> str:
    cmd = ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", f"{BASE_URL}/{path}"]
    if VERBOSE:
        print(f"$ {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()


# --------------------------------------------------------------------------
# Server lifecycle — auto-starts Chroma so you never type `chroma run`
# --------------------------------------------------------------------------

def ensure_server_running():
    if curl_status_code("heartbeat") == "200":
        print(f"Chroma server already running on port {PORT}.")
        return

    print(f"Starting Chroma server on port {PORT} in the background...")
    CHROMA_DATA_DIR.mkdir(exist_ok=True)
    log_file = open("chroma_server.log", "w")
    subprocess.Popen(
        ["chroma", "run", "--path", str(CHROMA_DATA_DIR), "--port", str(PORT)],
        stdout=log_file, stderr=log_file, start_new_session=True,
    )
    for _ in range(30):
        if curl_status_code("heartbeat") == "200":
            print("Server is up.")
            return
        time.sleep(1)
    raise RuntimeError("Chroma server did not start in time. Check chroma_server.log")


# --------------------------------------------------------------------------
# Collection actions — every one of these is a curl call
# --------------------------------------------------------------------------

def get_or_create_collection_id() -> str:
    resp = curl("POST", f"{TENANT_DB}/collections",
                {"name": COLLECTION_NAME, "get_or_create": True})
    collection_id = resp["id"]
    COLLECTION_ID_FILE.write_text(collection_id)
    return collection_id


def add_chunks(collection_id, ids, docs, embeddings, metadatas):
    curl("POST", f"{TENANT_DB}/collections/{collection_id}/add", {
        "ids": ids, "documents": docs, "embeddings": embeddings, "metadatas": metadatas,
    })


def count_chunks(collection_id: str) -> int:
    resp = curl("GET", f"{TENANT_DB}/collections/{collection_id}/count")
    return resp if isinstance(resp, int) else int(resp)


def query_chunks(collection_id: str, embedding: list[float], top_k: int) -> dict:
    return curl("POST", f"{TENANT_DB}/collections/{collection_id}/query", {
        "query_embeddings": [embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    })


# --------------------------------------------------------------------------
# PDF parsing + embedding (not network actions — curl cannot do these)
# --------------------------------------------------------------------------

_VECTORIZER = HashingVectorizer(n_features=384, alternate_sign=False, norm="l2")


def embed_texts(texts: list[str]) -> list[list[float]]:
    return _VECTORIZER.transform(texts).toarray().astype(np.float32).tolist()


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def build_chunks(pdf_dir: str):
    pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")
    ids, docs, metas = [], [], []
    for pdf_path in pdf_files:
        for page_number, page_text in extract_pages(pdf_path):
            for idx, piece in enumerate(chunk_text(page_text)):
                ids.append(f"{pdf_path.stem}_p{page_number}_c{idx}")
                docs.append(piece)
                metas.append({"source_file": pdf_path.name, "page_number": page_number})
    return ids, docs, metas


# --------------------------------------------------------------------------
# Generation step, gated by LLM_PROVIDER
# --------------------------------------------------------------------------

def generate_answer(question: str, matches: list[dict], provider: str) -> str:
    """
    provider == "none": no model call of any kind. Returns the retrieved
    chunks verbatim, clearly labelled, so it's obvious nothing was
    synthesized or rewritten.

    Any other value: not implemented in this build. Raises rather than
    silently degrading to retrieval-only, so a misconfigured provider
    name is never mistaken for a working one.
    """
    if provider == "none":
        if not matches:
            return "No sufficiently relevant passage found in the indexed PDFs."
        lines = [f"[retrieval-only, LLM_PROVIDER=none — verbatim source text below]\n"]
        for rank, m in enumerate(matches, start=1):
            lines.append(
                f"--- Match #{rank} | distance={m['distance']:.4f} | "
                f"source={m['source_file']} | page={m['page_number']} ---\n{m['text']}"
            )
        return "\n\n".join(lines)

    raise NotImplementedError(
        f"LLM_PROVIDER='{provider}' is not implemented in this build. "
        f"Use LLM_PROVIDER=none for retrieval-only mode."
    )


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_ingest(args):
    ensure_server_running()
    collection_id = get_or_create_collection_id()
    print(f"Collection ID: {collection_id}")

    ids, docs, metas = build_chunks(args.pdf_dir)
    print(f"Parsed {len(ids)} chunks from PDFs in '{args.pdf_dir}'.")

    BATCH = 50
    for i in range(0, len(ids), BATCH):
        batch_docs = docs[i:i + BATCH]
        add_chunks(collection_id, ids[i:i + BATCH], batch_docs,
                   embed_texts(batch_docs), metas[i:i + BATCH])
        print(f"  added chunks {i}-{i + len(batch_docs) - 1} via curl POST /add")

    print(f"Total chunks stored (via curl GET /count): {count_chunks(collection_id)}")


def cmd_query(args):
    ensure_server_running()
    if not COLLECTION_ID_FILE.exists():
        print("No collection found — run 'ingest' first.", file=sys.stderr)
        sys.exit(1)
    collection_id = COLLECTION_ID_FILE.read_text().strip()

    embedding = embed_texts([args.q])[0]
    result = query_chunks(collection_id, embedding, args.top_k)

    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]
    matches = [
        {"text": d, "source_file": m["source_file"], "page_number": m["page_number"], "distance": dist}
        for d, m, dist in zip(docs, metas, dists)
    ]

    provider = args.llm_provider or os.environ.get("LLM_PROVIDER", "none")
    answer = generate_answer(args.q, matches, provider)

    print(f"\nQUESTION: {args.q}\n" + "=" * 70)
    print(answer)


def main():
    global VERBOSE
    parser = argparse.ArgumentParser(description="PDF -> ChromaDB via curl, LLM_PROVIDER-gated answering")
    parser.add_argument("--verbose", action="store_true", help="Print every curl command run")
    parser.add_argument("--llm-provider", default=None,
                         help="Overrides LLM_PROVIDER env var. Default: 'none' (retrieval-only).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--pdf-dir", default="pdfs")
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = sub.add_parser("query")
    p_query.add_argument("--q", required=True)
    p_query.add_argument("--top-k", type=int, default=3)
    p_query.set_defaults(func=cmd_query)

    args = parser.parse_args()
    VERBOSE = args.verbose
    args.func(args)


if __name__ == "__main__":
    main()
