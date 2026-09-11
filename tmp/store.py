"""
store.py - Module for managing vector storage directly via Chroma client.
"""

import chromadb
from langchain_core.documents import Document


def get_chroma_client(host: str = "chromadb", port: int = 8000):
    """Creates an HTTP client pointing to the remote/Docker Chroma server."""
    return chromadb.HttpClient(host=host, port=port)


def create_vector_store(
    chunks: list[Document], host: str = "chromadb", port: int = 8000, collection_name: str = "rag_collection"
):
    """
    Stores document chunks directly into the remote Chroma server 
    using its native client (delegating embedding to the server or default lightweight setup).
    """
    client = get_chroma_client(host=host, port=port)
    
    # Re-create collection cleanly for learning/testing purposes
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
        
    collection = client.create_collection(name=collection_name)

    # Extract text, metadata, and generate unique IDs for each chunk
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"id_{i}" for i in range(len(chunks))]

    # Add texts directly to the Chroma server collection
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    return collection
