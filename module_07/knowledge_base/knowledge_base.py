"""
Module 07 — Personal Knowledge Base
Run with:
    .venv/bin/python module_07/knowledge_base/knowledge_base.py
(the ./chroma_db folder is created relative to the current working directory)
"""

import chromadb

# ── Client & collection ──────────────────────────────────────────────────────
# PersistentClient writes to disk, so data survives between runs.
client = chromadb.PersistentClient(path="./chroma_db")
# get_or_create_collection is idempotent: re-runs reuse the existing collection
# instead of raising "collection already exists".
collection = client.get_or_create_collection(name="my_knowledge")

# ── Documents ────────────────────────────────────────────────────────────────
# Three parallel lists: index i in each list describes the same document.
# Module numbers are strings because Chroma's where-filter matches exact types.
ids = [f"doc_{i:02d}" for i in range(1, 19)]

documents = [
    # Module 3 — databases
    "SQL joins combine rows from two tables based on a related column such as a foreign key.",
    "An index on a frequently queried column speeds up reads at the cost of slower writes.",
    "Database transactions group several statements so they either all commit or all roll back.",
    "Normalization splits data into related tables to avoid duplicated information.",
    # Module 5 — APIs
    "FastAPI builds REST endpoints from Python functions and generates OpenAPI docs automatically.",
    "Pydantic validates request data and rejects payloads that do not match the declared schema.",
    "HTTP status codes tell clients whether a request succeeded: 200 is OK, 404 is not found, 500 is a server error.",
    "API keys and tokens should be loaded from environment variables, never hard-coded in source.",
    "Rate limiting protects an API from being overwhelmed by too many requests from one client.",
    # Module 6 — frontend
    "React components re-render when their state or props change, keeping the UI in sync with data.",
    "CSS flexbox lays out items in a row or column and distributes space between them.",
    "The fetch API sends asynchronous HTTP requests from the browser and returns a promise.",
    "Responsive design uses media queries so a page adapts to phone, tablet and desktop widths.",
    # Module 7 — AI and embeddings
    "An embedding turns text into a vector so that similar meanings end up close together.",
    "Cosine distance measures the angle between two vectors; smaller values mean more similar text.",
    "ChromaDB stores embeddings and supports metadata filters alongside semantic search.",
    "Chunking splits long documents into smaller passages so retrieval returns focused context.",
    "Retrieval-augmented generation feeds retrieved passages to a language model to ground its answer.",
]

metadatas = [
    {"module": "3", "topic": "database"},
    {"module": "3", "topic": "database"},
    {"module": "3", "topic": "database"},
    {"module": "3", "topic": "database"},
    {"module": "5", "topic": "api"},
    {"module": "5", "topic": "api"},
    {"module": "5", "topic": "api"},
    {"module": "5", "topic": "api"},
    {"module": "5", "topic": "api"},
    {"module": "6", "topic": "frontend"},
    {"module": "6", "topic": "frontend"},
    {"module": "6", "topic": "frontend"},
    {"module": "6", "topic": "frontend"},
    {"module": "7", "topic": "ai"},
    {"module": "7", "topic": "ai"},
    {"module": "7", "topic": "ai"},
    {"module": "7", "topic": "ai"},
    {"module": "7", "topic": "ai"},
]

# Guard against the lists drifting out of sync when edited by hand.
assert len(ids) == len(documents) == len(metadatas), "parallel lists must match"

# upsert() inserts new IDs and overwrites existing ones, so re-running the script
# never fails with a duplicate-ID error (add() would) and never creates copies.
collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
print(f"Collection '{collection.name}' holds {collection.count()} documents.\n")


# ── Search function ──────────────────────────────────────────────────────────
def search(query: str, module_filter: str = None, n_results: int = 5):
    """Return top results for query, optionally filtered to a specific module."""
    # Empty/whitespace queries carry no meaning to embed; bail out early.
    if not query or not query.strip():
        print("  (empty query — nothing to search)")
        return None

    kwargs = {"query_texts": [query], "n_results": n_results}
    if module_filter is not None:
        # Metadata was stored as strings, so cast: where={"module": 5} would
        # match nothing because 5 != "5".
        kwargs["where"] = {"module": str(module_filter)}
    # `where` is only passed when non-empty; an empty dict is rejected by Chroma.

    results = collection.query(**kwargs)

    # Results are nested one list per query; we sent one query, hence [0].
    # `or [[]]` keeps the unpacking safe if a key is missing or empty.
    found_ids = (results.get("ids") or [[]])[0]
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]

    if not found_ids:
        print("  (no results)")
        return results

    for rank, (doc_id, dist, meta, doc) in enumerate(zip(found_ids, dists, metas, docs), start=1):
        # Lower distance = more similar.
        print(f"  {rank}. [{doc_id}] distance={dist:.4f} metadata={meta}")
        print(f"     {doc}")
    return results


# ── Test queries ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Broad search (no filter) ===")
    print("Query: 'how do computers understand meaning of text'")
    search("how do computers understand meaning of text")

    print("\n=== Filtered to a specific module ===")
    print("Query: 'sending requests' | module_filter='5'")
    search("sending requests", module_filter="5")

    print("\n=== Different words than stored documents ===")
    print("Query: 'how to handle bad user input'")
    search("how to handle bad user input")

    # Extra robustness checks
    print("\n=== Edge: module with no documents ===")
    search("databases", module_filter="99")

    print("\n=== Edge: empty / whitespace / special-character queries ===")
    for q in ["", "   ", "!@#$%^&*()", "'; DROP TABLE users; --"]:
        print(f"Query: {q!r}")
        search(q, n_results=2)

    print("\n=== Edge: n_results larger than matching docs, int filter ===")
    search("frontend layout", module_filter=6, n_results=50)
