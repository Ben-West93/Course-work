"""
L4 — Reliability & Guardrails: Enhanced RAG Pipeline  (STARTER)
================================================================
Run with:
    python rag_pipeline.py

Prerequisites:
    pip install chromadb requests
    Ollama running with a model pulled.
    A docs/ folder populated (reuse from Lesson 3 / building-the-rag-pipeline).

Your goal: extend the basic RAG pipeline with 4 production guardrails.

Guardrails to add:
    1. Distance threshold filtering — discard chunks with distance >= THRESHOLD
       before passing them to the LLM; return a "no relevant info" response if
       all chunks are filtered out
    2. Confidence levels — compute "high" / "medium" / "low" from the best
       (lowest) distance among the kept chunks
    3. Strengthened system prompt — explicitly forbid making up information,
       require "I don't know" for uncertain answers, require source citation
    4. Structured response — return a dict instead of a plain string:
           {"answer": str, "sources": list[str], "confidence": str,
            "chunks_retrieved": int}

Key concepts:
    Distance threshold:
        ChromaDB uses L2 distance — lower = more similar.
        A distance of 0.0 is a perfect match; ~1.5 is quite dissimilar.
        Start with DISTANCE_THRESHOLD = 1.0 and tune from there.

    Confidence mapping:
        best_distance = min(chunk["distance"] for chunk in kept_chunks)
        "high"   if best_distance < 0.5
        "medium" if best_distance < 1.0
        "low"    otherwise

    Structured return:
        return {
            "answer": "...",
            "sources": ["doc1.txt", "doc2.md"],   # unique source filenames
            "confidence": "high",
            "chunks_retrieved": 3,
        }
"""

import chromadb
import requests
import json
import os

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2:1b"
DISTANCE_THRESHOLD = 1.0   # Guardrail 1 — tune as needed


# ── Guardrail 3: Strengthened system prompt ────────────────────────────────
# TODO: Write a SYSTEM_PROMPT that:
#   - Instructs the model to answer ONLY from the provided context
#   - Explicitly says: never make up information
#   - Requires "I don't know" when the answer isn't in the context
#   - Requires source citation by document name
SYSTEM_PROMPT = ""  # TODO


# ── Core pipeline functions ────────────────────────────────────────────────
# (Copy / adapt from your my_rag.py from Lesson 3)

def load_documents(directory: str) -> list[dict]:
    """Load .txt/.md files, chunk by paragraph."""
    pass  # TODO: copy from my_rag.py


def ingest(collection, docs_directory: str) -> int:
    """Upsert all chunks into ChromaDB."""
    pass  # TODO: copy from my_rag.py


def retrieve(collection, query: str, n_results: int = 3) -> list[dict]:
    """Query ChromaDB, return list of {"text", "metadata", "distance"}."""
    pass  # TODO: copy from my_rag.py


def generate(messages: list[dict]) -> str:
    """Call Ollama (non-streaming for structured use). Return response text."""
    # TODO: POST to Ollama with stream=False
    # TODO: Return response.json()["message"]["content"]
    # TODO: Handle ConnectionError gracefully
    pass  # TODO


# ── Guardrail 1 + 2: Filter and score ─────────────────────────────────────

def filter_chunks(chunks: list[dict], threshold: float = DISTANCE_THRESHOLD) -> list[dict]:
    """Guardrail 1 — discard chunks with distance >= threshold."""
    # TODO: return [c for c in chunks if c["distance"] < threshold]
    pass  # TODO


def compute_confidence(chunks: list[dict]) -> str:
    """Guardrail 2 — return 'high', 'medium', or 'low' based on best distance."""
    # TODO: Find the minimum distance among chunks
    # TODO: Return "high" if < 0.5, "medium" if < 1.0, else "low"
    pass  # TODO


# ── Guardrail 4: Structured RAG query ─────────────────────────────────────

def rag_query(collection, question: str) -> dict:
    """Full guardrailed pipeline. Returns a structured result dict."""
    # TODO: retrieve() — get raw chunks
    # TODO: filter_chunks() — apply distance threshold
    # TODO: If no chunks remain, return:
    #   {"answer": "I couldn't find relevant information ...",
    #    "sources": [], "confidence": "low", "chunks_retrieved": 0}
    # TODO: compute_confidence() on kept chunks
    # TODO: Build messages with SYSTEM_PROMPT + context
    # TODO: generate() to get answer
    # TODO: Collect unique source filenames
    # TODO: Return structured dict
    pass  # TODO


# ── Main: test with 4 query types ─────────────────────────────────────────

if __name__ == "__main__":
    client = chromadb.PersistentClient(path="./rag_db")
    collection = client.get_or_create_collection("course_docs")

    if collection.count() == 0:
        count = ingest(collection, "docs")
        print(f"Ingested {count} chunks.")
    else:
        print(f"Using existing collection: {collection.count()} chunks.")

    # TODO: Define 4 test questions (in-scope, partial, out-of-scope, ambiguous)
    test_questions = [
        # ("in-scope",     "..."),
        # ("partial",      "..."),
        # ("out-of-scope", "..."),
        # ("ambiguous",    "..."),
    ]

    for label, question in test_questions:
        print(f"\n{'='*60}")
        print(f"[{label.upper()}] {question}")
        result = rag_query(collection, question)
        print(f"Confidence   : {result['confidence']}")
        print(f"Chunks used  : {result['chunks_retrieved']}")
        print(f"Sources      : {result['sources']}")
        print(f"Answer       :\n{result['answer']}")
