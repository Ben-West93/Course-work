"""
Building the RAG Pipeline
Run with:
    python my_rag.py

Ingests docs/ into a persistent ChromaDB collection, retrieves the top 3
chunks for each question and streams an answer from Ollama.
"""

import chromadb
import requests
import json
import os
import time

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2:1b"  # Adjust to your downloaded model

# resolve paths from the script so it works no matter where it's run from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
DB_DIR = os.path.join(BASE_DIR, "rag_db")


# ── Step 1: Ingestion ──────────────────────────────────────────────────────

def load_documents(directory: str) -> list[dict]:
    """Load all .txt and .md files from directory, chunk by paragraph."""
    if not os.path.isdir(directory):
        print(f"Warning: docs folder not found: {directory}")
        return []

    chunks = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith((".txt", ".md")):
            continue
        with open(os.path.join(directory, filename), encoding="utf-8", errors="replace") as f:
            text = f.read().replace("\r\n", "\n")

        # blank lines separate paragraphs, so each paragraph becomes one chunk.
        # empty pieces (e.g. from trailing newlines) are dropped before numbering
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for i, para in enumerate(paragraphs):
            chunks.append({
                "text": para,
                "id": f"{filename}_{i}",
                "metadata": {"source": filename, "chunk_index": str(i)},
            })
    return chunks


def ingest(collection, docs_directory: str) -> int:
    """Chunk and store all documents in ChromaDB. Returns chunk count."""
    chunks = load_documents(docs_directory)
    if not chunks:
        print(f"Warning: no .txt or .md content found in {docs_directory}")
        return 0

    # upsert instead of add so re-running doesn't fail on existing ids
    collection.upsert(
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        ids=[c["id"] for c in chunks],
    )
    return len(chunks)


# ── Step 2: Retrieval ──────────────────────────────────────────────────────

def retrieve(collection, query: str, n_results: int = 3) -> list[dict]:
    """Query ChromaDB, return top chunks as list of dicts."""
    count = collection.count()
    if count == 0:
        return []

    # chroma errors if n_results is bigger than the collection
    results = collection.query(query_texts=[query], n_results=min(n_results, count))

    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


# ── Step 3: Prompt building ────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful AI assistant.
Answer the user's question based ONLY on the context provided below.

Rules:
- Only use information from the CONTEXT section. Do not use outside knowledge.
- If the context doesn't contain the answer, reply exactly: "I don't have enough information to answer that."
- Cite your sources by mentioning the document name in parentheses, e.g. (chromadb.md).
- Keep your response under 200 words.
"""


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    """Build the RAG messages list: system (with context) + user question."""
    context = "\n\n---\n\n".join(
        f"[Source: {c['metadata']['source']}]\n{c['text']}" for c in chunks
    )
    system = f"{SYSTEM_PROMPT}\nCONTEXT:\n{context or '(no context found)'}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


# ── Step 4: Generation ─────────────────────────────────────────────────────

def generate(messages: list[dict]) -> str:
    """Stream a response from Ollama, return the full text."""
    payload = {"model": MODEL, "messages": messages, "stream": True}
    answer = ""
    try:
        with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            # ollama sends one JSON object per line, each holding the next token
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = data.get("message", {}).get("content", "")
                print(token, end="", flush=True)
                answer += token
                if data.get("done"):
                    break
        print()
    except requests.exceptions.ConnectionError:
        # server isn't up - return a message instead of crashing the loop
        msg = f"Error: can't reach Ollama at {OLLAMA_URL}. Start it with `ollama serve`."
        print(msg)
        return msg
    except requests.exceptions.HTTPError as e:
        # usually means the model hasn't been pulled
        msg = f"Error: Ollama returned {e.response.status_code}. Try `ollama pull {MODEL}`."
        print(msg)
        return msg
    return answer


# ── Step 5: Full RAG query ─────────────────────────────────────────────────

def rag_query(collection, question: str) -> str:
    """Retrieve → build prompt → generate. Prints retrieved chunks first."""
    print(f"\n{'=' * 60}\nQ: {question}\n{'=' * 60}")
    start = time.time()

    chunks = retrieve(collection, question)
    print(f"Retrieved {len(chunks)} chunks:")
    for i, c in enumerate(chunks, 1):
        meta = c["metadata"]
        print(f"  [{i}] {meta['source']} #{meta['chunk_index']}  (distance: {c['distance']:.3f})")

    print("\nAnswer: ", end="", flush=True)
    answer = generate(build_messages(question, chunks))
    print(f"\n({time.time() - start:.1f}s)")
    return answer


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # PersistentClient saves to disk, so embeddings only get built on the first run
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection("course_docs")

    if collection.count() == 0:
        print(f"Ingesting documents from {DOCS_DIR} ...")
        n = ingest(collection, DOCS_DIR)
        print(f"Ingested {n} chunks.")
    else:
        print(f"Collection already has {collection.count()} chunks, skipping ingest.")

    print(f"\nReady. Using {MODEL}. Type 'quit' to exit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        rag_query(collection, question)

    print("Bye.")
