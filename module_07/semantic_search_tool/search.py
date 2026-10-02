"""
Module 7 Project — Semantic Search Tool
========================================
search.py — query ChromaDB and return ranked results

Import this module into app.py:
    from search import search, get_collection_stats
"""

import chromadb

# ── Configuration (shared with ingest.py so the two can never drift apart) ────
from ingest import CHROMA_PATH, COLLECTION_METADATA, COLLECTION_NAME, embed

# Score cut-offs for the relevance indicator (score = cosine similarity).
HIGH_RELEVANCE   = 0.50
MEDIUM_RELEVANCE = 0.30


def get_collection():
    """Return the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata=COLLECTION_METADATA
    )


def is_searchable(query: str) -> bool:
    """
    Return True if the query contains at least one letter or digit.

    Empty, whitespace-only, and punctuation/emoji-only queries carry no meaning
    for the embedding model, so they are treated as empty.
    """
    return bool(query) and any(ch.isalnum() for ch in query)


def relevance_label(score: float) -> str:
    """Map a similarity score to a human-readable relevance indicator."""
    if score >= HIGH_RELEVANCE:
        return "High"
    if score >= MEDIUM_RELEVANCE:
        return "Medium"
    return "Low"


def search(
    query: str,
    n_results: int = 5,
    sources: list[str] = None,
    distance_threshold: float = None,
) -> list[dict]:
    """
    Search the ChromaDB collection and return ranked results.

    Args:
        query:              Natural language search query.
        n_results:          Maximum number of results to return.
        sources:            If provided, only return chunks from these filenames.
        distance_threshold: If provided, exclude results with distance above this
                            value (lower = more similar).

    Returns:
        List of result dicts sorted by distance ascending (best first):
            {
                "text":        str,
                "source":      str,
                "chunk_index": int,
                "distance":    float,
                "score":       float,  # 1 - distance
                "relevance":   str,    # "High" / "Medium" / "Low"
            }
        Returns [] for empty / punctuation-only queries, n_results < 1, or if
        the collection has no documents.
    """
    if not is_searchable(query) or n_results < 1:
        return []

    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []

    query_args = {
        "query_embeddings": embed([query.strip()]),
        "n_results": min(n_results, total),
        "include": ["documents", "metadatas", "distances"],
    }
    if sources:
        query_args["where"] = {"source": {"$in": list(sources)}}

    raw = collection.query(**query_args)

    results = []
    for text, meta, distance in zip(
        raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
    ):
        if distance_threshold is not None and distance > distance_threshold:
            continue
        score = 1 - distance
        results.append({
            "text": text,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "distance": round(distance, 4),
            "score": round(score, 4),
            "relevance": relevance_label(score),
        })

    return sorted(results, key=lambda r: r["distance"])


def get_collection_stats() -> dict:
    """
    Return basic stats about the indexed collection.

    Returns:
        {
            "total_chunks":   int,
            "unique_sources": int,
            "source_names":   list[str],
            "chunk_size":     int | None,  # chunk size used for the current index
            "overlap":        int | None,
        }
    """
    collection = get_collection()
    metadatas = collection.get(include=["metadatas"])["metadatas"] or []
    source_names = sorted({m["source"] for m in metadatas})
    first = metadatas[0] if metadatas else {}
    return {
        "total_chunks": len(metadatas),
        "unique_sources": len(source_names),
        "source_names": source_names,
        "chunk_size": first.get("chunk_size"),
        "overlap": first.get("overlap"),
    }
