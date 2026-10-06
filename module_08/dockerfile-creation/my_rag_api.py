"""
Connecting RAG to FastAPI
Run from this folder with:
    uvicorn my_rag_api:app --reload --port 8000

or in Docker (see README.md):
    docker run -d -p 8000:8000 --name rag-container my-rag-api

Then open http://localhost:8000/docs to test the endpoints in Swagger UI.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import chromadb
import requests
import os

app = FastAPI(title="RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ─────────────────────────────────────────────────────────────────
# inside a container localhost is the container itself, so pass
# -e OLLAMA_URL=http://host.docker.internal:11434 to reach Ollama on the host
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = "llama3.2:1b"

# anchored to this file so it works no matter where uvicorn is started from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "rag_db")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# ── ChromaDB setup ─────────────────────────────────────────────────────────
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection("documents")


# ── Pydantic schemas ───────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    n_results: int = 3
    max_distance: float = 1.2

    # str alone accepts "" and "   ", so check it here. A ValueError inside a
    # validator becomes a 422 response before the route even runs
    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        return v


class SourceChunk(BaseModel):
    text: str
    source: str
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    confidence: str


class IngestResponse(BaseModel):
    chunks_ingested: int
    message: str


# ── RAG helpers ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer ONLY from the provided context. "
    "If the context doesn't contain the answer, say you don't have enough "
    "information. Cite source documents by name. Keep responses under 200 words."
)

NO_DOCS_ANSWER = "No relevant documents found. Try /ingest first or rephrase the question."


def load_documents(directory: str) -> list[dict]:
    if not os.path.isdir(directory):
        return []

    chunks = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith((".txt", ".md")):
            continue
        with open(os.path.join(directory, filename), encoding="utf-8", errors="replace") as f:
            text = f.read().replace("\r\n", "\n")

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for i, para in enumerate(paragraphs):
            chunks.append({
                "text": para,
                "id": f"{filename}_{i}",
                "metadata": {"source": filename, "chunk_index": i},
            })
    return chunks


def retrieve(query: str, n_results: int = 3, max_distance: float = 1.2) -> list[dict]:
    count = collection.count()
    if count == 0:
        return []

    # chroma errors if n_results is 0 or more than what's stored
    n = max(1, min(n_results, count))
    results = collection.query(query_texts=[query], n_results=n)

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        # lower distance = closer match. Anything past the cutoff is more
        # likely noise than context, so it never reaches the prompt
        if dist > max_distance:
            continue
        chunks.append({"text": doc, "source": meta["source"], "distance": dist})
    return chunks


def compute_confidence(chunks: list[dict]) -> str:
    if not chunks:
        return "low"
    best = min(c["distance"] for c in chunks)
    if best < 0.5:
        return "high"
    if best < 0.8:
        return "medium"
    return "low"


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    context = "\n\n---\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)
    return [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"},
        {"role": "user", "content": question},
    ]


def call_ollama(messages: list[dict]) -> str:
    payload = {"model": MODEL, "messages": messages, "stream": False,
               "options": {"temperature": 0}}
    # 503 rather than 500: the API itself is fine, the service it depends on
    # isn't, and the client can retry once Ollama is back up
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama service is unavailable")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=503, detail="Ollama timed out")

    # e.g. 404 when the model hasn't been pulled yet
    if r.status_code != 200:
        raise HTTPException(status_code=503, detail=f"Ollama returned {r.status_code} for model {MODEL}")
    return r.json()["message"]["content"].strip()


def check_ollama_health() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    chunks = retrieve(req.question, req.n_results, req.max_distance)

    # empty collection or nothing within max_distance, skip the LLM call
    if not chunks:
        return AskResponse(answer=NO_DOCS_ANSWER, sources=[], confidence="low")

    answer = call_ollama(build_messages(req.question, chunks))
    return AskResponse(
        answer=answer,
        sources=[SourceChunk(**c) for c in chunks],
        confidence=compute_confidence(chunks),
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    chunks = load_documents(DOCS_DIR)
    if not chunks:
        return IngestResponse(
            chunks_ingested=0,
            message=f"No .txt or .md content found in {DOCS_DIR}",
        )

    # stable ids, so re-ingesting overwrites instead of duplicating
    collection.upsert(
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        ids=[c["id"] for c in chunks],
    )
    files = len({c["metadata"]["source"] for c in chunks})
    return IngestResponse(
        chunks_ingested=len(chunks),
        message=f"Ingested {len(chunks)} chunks from {files} files",
    )


@app.get("/stats")
def stats():
    return {"document_count": collection.count(), "model": MODEL, "db_path": DB_PATH}


@app.get("/health")
def health():
    # chroma: if count() works the db is open and readable
    try:
        document_count = collection.count()
        chroma_ok = True
    except Exception:
        document_count = None
        chroma_ok = False

    # ollama: /api/tags is cheap and doesn't load a model
    ollama_ok = check_ollama_health()

    if chroma_ok and ollama_ok:
        status = "ok"
    elif chroma_ok:
        status = "degraded"  # can still ingest and retrieve, just can't answer
    else:
        status = "error"

    return {
        "status": status,
        "chromadb": "ok" if chroma_ok else "error",
        "ollama": "connected" if ollama_ok else "disconnected",
        "document_count": document_count,
        "model": MODEL,
    }
