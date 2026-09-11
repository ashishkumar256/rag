"""
main.py - Entry point for parsing, chunking, local embedding, and storing in ChromaDB.
"""

import os
from ingest import chunk_documents, load_and_clean_pdf
from store import create_vector_store

# ------------------------------------------------------------------
# Setup Configurations
# ------------------------------------------------------------------
PDF_FILE_PATH = "/data/pdfs/01_AWS_Basic_Foundations_Security.pdf"
VECTORDB_HOST = os.getenv("VECTORDB_HOST", "chromadb")
VECTORDB_PORT = int(os.getenv("VECTORDB_PORT", 8000))


def main():
    print("1. Parsing and cleaning PDF...")
    documents = load_and_clean_pdf(PDF_FILE_PATH, header_margin=50, footer_margin=50)
    print(f"   Loaded {len(documents)} clean pages.")

    print("2. Chunking document text...")
    chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=200)
    print(f"   Generated {len(chunks)} total text chunks.")

    print("3. Generating local embeddings and storing in remote Chroma server...")
    create_vector_store(chunks, host=VECTORDB_HOST, port=VECTORDB_PORT)
    print("   Vector database populated successfully!")


if __name__ == "__main__":
    main()
