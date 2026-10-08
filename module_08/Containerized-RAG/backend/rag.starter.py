"""
Module 8 Project — Containerized RAG Assistant  (STARTER)
Backend: RAG Pipeline
======================
This is the core of your RAG application.
Implement each function below, then wire them into main.py.

Pipeline overview:
    1. load_documents(directory)  — read .txt / .md files and split into chunks
    2. retrieve(query, n)         — query ChromaDB for the most relevant chunks
    3. build_prompt(question, chunks) — format system prompt + context
    4. generate(messages)         — call Ollama's /api/chat endpoint
    5. compute_confidence(chunks) — rate result quality from distances

Run locally to test before wiring to FastAPI:
    python rag.py
"""

import os
import requests
import chromadb
from config import settings

# ── ChromaDB client ────────────────────────────────────────────────────────
# TODO: Create a PersistentClient pointed at settings.chroma_path
# chroma_client = chromadb.PersistentClient(path=...)
# collection    = chroma_client.get_or_create_collection("documents")


# ── 1. Document loading ────────────────────────────────────────────────────

def load_documents(directory: str) -> list[dict]:
    """
    Read all .txt and .md files in `directory`.
    Split each file into paragraphs (split on blank lines).
    Return a list of dicts with keys: "text", "id", "metadata".

    Each chunk dict should look like:
        {
            "text":     "the paragraph text",
            "id":       "filename_0",          # unique id per chunk
            "metadata": {"source": "filename", "chunk_index": "0"},
        }

    TODO: Implement this function.
    Steps:
      1. os.listdir(directory) — iterate over files ending in .txt or .md
      2. Read each file and split on "\\n\\n"
      3. Build a dict for each non-empty paragraph
      4. Return the list
    """
    # TODO: implement
    return []


# ── 2. Retrieval ───────────────────────────────────────────────────────────

def retrieve(query: str, n_results: int = 3, max_distance: float = 1.2) -> list[dict]:
    """
    Query ChromaDB for the `n_results` most relevant chunks.
    Filter out any chunks with distance > max_distance.
    Return a list of dicts with keys: "text", "metadata", "distance".

    TODO: Implement this function.
    Steps:
      1. Check collection.count() — return [] if empty
      2. collection.query(query_texts=[query], n_results=...)
      3. Zip documents, metadatas, and distances into result dicts
      4. Filter by max_distance
    """
    # TODO: implement
    return []


# ── 3. Prompt building ─────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer ONLY from the provided context. "
    "If the context doesn't contain the answer, say you don't have enough "
    "information. Cite source documents by name. Keep responses under 200 words."
)


def build_prompt(question: str, chunks: list[dict]) -> list[dict]:
    """
    Build the messages list for Ollama's /api/chat endpoint.
    Format:
        [
            {"role": "system", "content": "<system prompt>\\n\\nCONTEXT:\\n<context>"},
            {"role": "user",   "content": "<question>"},
        ]

    TODO: Implement this function.
    Steps:
      1. Join chunks into a context string, labelling each with its source filename
      2. Build the system message combining SYSTEM_PROMPT + context
      3. Return [system_message, user_message]
    """
    # TODO: implement
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]


# ── 4. Generation ──────────────────────────────────────────────────────────

def generate(messages: list[dict]) -> str:
    """
    Send messages to Ollama's /api/chat endpoint and return the response text.
    Uses settings.ollama_url and settings.model_name.

    TODO: Implement this function.
    Steps:
      1. requests.post(settings.ollama_url + "/api/chat", json={...})
      2. Payload: {"model": settings.model_name, "messages": messages, "stream": False}
      3. Parse response.json()["message"]["content"]
      4. Handle requests.exceptions.ConnectionError gracefully
    """
    # TODO: implement
    return "Ollama is not connected — implement generate() in rag.py"


# ── 5. Confidence scoring ──────────────────────────────────────────────────

def compute_confidence(chunks: list[dict]) -> str:
    """
    Return "high", "medium", or "low" based on the best (lowest) distance score.

    Suggested thresholds (adjust to taste):
        distance < 0.5  → "high"
        distance < 1.0  → "medium"
        otherwise       → "low"

    Return "low" if chunks is empty.

    TODO: Implement this function.
    """
    # TODO: implement
    return "low"


# ── Quick test (run as a script) ───────────────────────────────────────────

if __name__ == "__main__":
    docs = load_documents("./docs")
    print(f"Loaded {len(docs)} chunks from ./docs")

    if docs:
        # Ingest into ChromaDB
        # collection.upsert(
        #     documents=[d["text"] for d in docs],
        #     metadatas=[d["metadata"] for d in docs],
        #     ids=[d["id"] for d in docs],
        # )
        print("TODO: upsert docs into collection, then test retrieval")
