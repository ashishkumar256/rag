"""
store.py - Module for managing vector storage using real local embeddings via FastEmbed.
"""

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import FastEmbedEmbeddings


def get_chroma_client(host: str = "chromadb", port: int = 8000):
    """Creates an HTTP client pointing to the remote/Docker Chroma server."""
    return chromadb.HttpClient(host=host, port=port)


def get_embedding_model():
    """Returns a real, lightweight local embedding model using FastEmbed."""
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def create_vector_store(
    chunks: list[Document], host: str = "chromadb", port: int = 8000, collection_name: str = "rag_collection"
):
    """
    Generates real local embeddings and stores document chunks into the remote Chroma server.
    """
    embedding_model = get_embedding_model()
    client = get_chroma_client(host=host, port=port)

    # Re-create collection cleanly for learning/testing purposes
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    # Use LangChain's Chroma wrapper to store documents with real embeddings
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        client=client,
        collection_name=collection_name,
    )
    return vectorstore
