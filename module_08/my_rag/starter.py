"""
L3 — Building the RAG Pipeline  (STARTER)
==========================================
Run with:
    python my_rag.py

Prerequisites:
    pip install chromadb requests
    Ollama running (`ollama serve`) with a model pulled.
    A docs/ folder containing at least 6 .txt or .md files.

Your goal: build a complete RAG pipeline — ingest → retrieve → generate.

Required features:
    1. load_documents(directory) — load .txt/.md files, chunk by paragraph,
       return list of {"text", "id", "metadata"} dicts
    2. ingest(collection, docs_directory) — upsert all chunks into ChromaDB,
       return chunk count
    3. retrieve(collection, query, n_results=3) — query ChromaDB, return list
       of {"text", "metadata", "distance"} dicts
    4. build_messages(question, chunks) — assemble RAG prompt with a system
       prompt that instructs citation, plus the retrieved context
    5. generate(messages) — POST to Ollama /api/chat (stream=True),
       print tokens as they arrive, return the full response string;
       handle ConnectionError gracefully
    6. rag_query(collection, question) — wire up retrieve → build_messages →
       generate; print retrieved chunks before the answer
    7. A __main__ block with a persistent ChromaDB client, ingest-on-first-run
       logic, and an interactive loop that exits on "quit"

Key concepts:
    ChromaDB persistent client:
        import chromadb
        client = chromadb.PersistentClient(path="./rag_db")
        collection = client.get_or_create_collection("course_docs")

    Upsert (add or update):
        collection.upsert(documents=[...], metadatas=[...], ids=[...])

    Query:
        results = collection.query(query_texts=[question], n_results=3)
        # results["documents"][0]  → list of chunk texts
        # results["metadatas"][0]  → list of metadata dicts
        # results["distances"][0]  → list of distance floats

    Streaming from Ollama:
        response = requests.post(url, json=payload, stream=True)
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                print(token, end="", flush=True)
"""

import chromadb
import requests
import json
import os
import time

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2:1b"  # Adjust to your downloaded model


# ── Step 1: Ingestion ──────────────────────────────────────────────────────

def load_documents(directory: str) -> list[dict]:
    """Load all .txt and .md files from directory, chunk by paragraph."""
    # TODO: Iterate sorted os.listdir(directory)
    # TODO: For each .txt / .md file, open and read it
    # TODO: Split content by "\n\n" to get paragraphs, strip each one
    # TODO: For each non-empty paragraph, append:
    #   {"text": para, "id": f"{filename}_{i}", "metadata": {"source": filename, "chunk_index": str(i)}}
    # TODO: Return the list
    pass  # TODO


def ingest(collection, docs_directory: str) -> int:
    """Chunk and store all documents in ChromaDB. Returns chunk count."""
    # TODO: Call load_documents()
    # TODO: If no chunks, print a warning and return 0
    # TODO: collection.upsert(documents=..., metadatas=..., ids=...)
    # TODO: Return len(chunks)
    pass  # TODO


# ── Step 2: Retrieval ──────────────────────────────────────────────────────

def retrieve(collection, query: str, n_results: int = 3) -> list[dict]:
    """Query ChromaDB, return top chunks as list of dicts."""
    # TODO: collection.query(query_texts=[query], n_results=min(n_results, collection.count()))
    # TODO: Build and return a list of {"text", "metadata", "distance"} dicts
    pass  # TODO


# ── Step 3: Prompt building ────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful AI assistant.
Answer the user's question based ONLY on the context provided below.

Rules:
- Only use information from the CONTEXT section.
- If the context doesn't contain the answer, say "I don't have enough information to answer that."
- Cite your sources by mentioning the document name in parentheses.
- Keep your response under 200 words.
"""


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    """Build the RAG messages list: system (with context) + user question."""
    # TODO: Format each chunk as "[Source: <source>]\n<text>"
    # TODO: Join chunks with "\n\n---\n\n"
    # TODO: Append context to SYSTEM_PROMPT
    # TODO: Return [{"role": "system", "content": ...}, {"role": "user", "content": question}]
    pass  # TODO


# ── Step 4: Generation ─────────────────────────────────────────────────────

def generate(messages: list[dict]) -> str:
    """Stream a response from Ollama, return the full text."""
    # TODO: POST to f"{OLLAMA_URL}/api/chat" with stream=True
    # TODO: Iterate iter_lines(), parse JSON, print each token, accumulate
    # TODO: Catch requests.exceptions.ConnectionError and return a friendly error message
    pass  # TODO


# ── Step 5: Full RAG query ─────────────────────────────────────────────────

def rag_query(collection, question: str) -> str:
    """Retrieve → build prompt → generate. Prints retrieved chunks first."""
    # TODO: Print a header with the question
    # TODO: Call retrieve(), print each chunk (index, distance, source)
    # TODO: Call build_messages(), then generate()
    # TODO: Print elapsed time, return the answer
    pass  # TODO


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # TODO: Create a PersistentClient at "./rag_db"
    # TODO: get_or_create_collection("course_docs")
    # TODO: If collection.count() == 0, call ingest(collection, "docs")
    # TODO: Otherwise print how many chunks already exist

    # TODO: Print a "Ready" message and start an interactive loop:
    #   - input("You: ")
    #   - break on "quit" / "exit" / "q"
    #   - call rag_query() on non-empty input
    pass  # TODO
