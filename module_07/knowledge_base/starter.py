"""
L7 — Personal Knowledge Base  (STARTER)
========================================
Run with:
    python starter.py

Your goal: build a persistent ChromaDB knowledge base from course content and
demonstrate metadata-filtered semantic search.

Required features:
    1. A persistent ChromaDB client (data survives restarts)
    2. A collection called "my_knowledge" created with get_or_create_collection
    3. At least 15 documents from at least 3 different modules, each with:
       - A unique ID (e.g. "doc_01")
       - Text content (a sentence or short paragraph)
       - Metadata with at least `module` (e.g. "5") and `topic`
         (e.g. "api", "database", "frontend", "ai")
    4. A search() function that accepts a query and optional module filter
       and returns the top 5 results with distances and metadata
    5. At least 3 test queries:
       - One broad (no filter)
       - One filtered to a specific module
       - One where the query words differ from the stored documents

Key concepts:
    Persistent client:
        client = chromadb.PersistentClient(path="./chroma_db")

    Upsert avoids errors on re-runs:
        collection.upsert(ids=[...], documents=[...], metadatas=[...])

    Metadata filtering:
        collection.query(
            query_texts=["your query"],
            n_results=5,
            where={"module": "5"}   # only return docs from module 5
        )

    Result structure:
        results["ids"][0]        # list of IDs
        results["documents"][0]  # list of document strings
        results["metadatas"][0]  # list of metadata dicts
        results["distances"][0]  # list of distances (lower = more similar)
"""

import chromadb

# ── Client & collection ──────────────────────────────────────────────────────
# TODO: Create a PersistentClient with path="./chroma_db"
# TODO: Get or create a collection called "my_knowledge"

# ── Documents ────────────────────────────────────────────────────────────────
# TODO: Define three lists that match in length:
#   ids       — unique strings like "doc_01", "doc_02", ...
#   documents — text summaries from at least 3 different modules
#   metadatas — dicts with at least {"module": "X", "topic": "Y"}
#
# Aim for at least 15 documents.

ids = [
    # TODO
]

documents = [
    # TODO — one sentence or short paragraph per document
]

metadatas = [
    # TODO — {"module": "1", "topic": "ai"}, {"module": "5", "topic": "api"}, ...
]

# TODO: Upsert all documents into the collection

# ── Search function ──────────────────────────────────────────────────────────
def search(query: str, module_filter: str = None, n_results: int = 5):
    """Return top results for query, optionally filtered to a specific module."""
    # TODO: Build the where dict only when module_filter is provided
    # TODO: Call collection.query() with query_texts, n_results, and where
    # TODO: Print each result's rank, distance, metadata, and document text
    pass


# ── Test queries ─────────────────────────────────────────────────────────────
print("=== Broad search (no filter) ===")
# TODO: Call search() with a broad query

print("\n=== Filtered to a specific module ===")
# TODO: Call search() with a query and a module_filter

print("\n=== Different words than stored documents ===")
# TODO: Call search() with a query that paraphrases rather than copies
#       the stored text — e.g. "how to handle bad user input" instead of
#       "Pydantic validates request data"
