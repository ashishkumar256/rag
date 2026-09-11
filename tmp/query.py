"""
query.py - Script for querying the remote ChromaDB vector store and getting raw chunk outputs.
"""

import os
from langchain_chroma import Chroma
from store import get_chroma_client, get_embedding_model

# ------------------------------------------------------------------
# Setup Configurations
# ------------------------------------------------------------------
VECTORDB_HOST = os.getenv("VECTORDB_HOST", "chromadb")
VECTORDB_PORT = int(os.getenv("VECTORDB_PORT", 8000))


def main():
    print("Connecting to remote Chroma server...")
    embedding_model = get_embedding_model()
    client = get_chroma_client(host=VECTORDB_HOST, port=VECTORDB_PORT)
    
    vectorstore = Chroma(
        client=client,
        collection_name="rag_collection",
        embedding_function=embedding_model,
    )

    # Change this query anytime to test different searches
    user_query = "Which S3 bucket for CloudTrail?"
    print(f"\nQuerying: '{user_query}'\n" + "-" * 50)

    # Fetch raw documents along with distance/similarity scores directly
    results = vectorstore.similarity_search_with_score(user_query, k=3)

    print("\n--- Raw Retrieval Results ---")
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n[Result {i}] (Distance Score: {score:.4f})")
        print(f"Metadata: {doc.metadata}")
        print(f"Content:\n{doc.page_content}")
        print("-" * 50)


if __name__ == "__main__":
    main()
