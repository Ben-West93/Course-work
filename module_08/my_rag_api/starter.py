"""
L5 — Connecting RAG to FastAPI  (STARTER)
==========================================
Run with:
    uvicorn my_rag_api:app --reload --port 8000

Then open http://localhost:8000/docs to test all endpoints in Swagger UI.

Prerequisites:
    pip install fastapi uvicorn chromadb requests pydantic
    A docs/ folder with text files.
    Ollama running (or handle the 503 case gracefully).

Your goal: wrap the RAG pipeline in a FastAPI application.

Endpoints to implement:
    POST /ask     — accept AskRequest, return AskResponse
    POST /ingest  — load docs/ into ChromaDB, return IngestResponse
    GET  /stats   — return document count + model name
    GET  /health  — check ChromaDB and Ollama, return status dict

Key concepts:
    CORS middleware (add before any routes):
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware,
            allow_origins=["*"], allow_credentials=True,
            allow_methods=["*"], allow_headers=["*"])

    Pydantic validation:
        class AskRequest(BaseModel):
            question: str          # Pydantic raises 422 for empty string
                                   # only if you add a validator — add one!
            n_results: int = 3
            max_distance: float = 1.2

    HTTPException for 503:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Ollama is not running")

    Checking Ollama liveness:
        GET http://localhost:11434/api/tags  — returns 200 if Ollama is up
        Catch requests.exceptions.ConnectionError to detect it's down
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import chromadb
import requests
import os

app = FastAPI(title="RAG API", version="1.0")

# TODO: Add CORSMiddleware — allow all origins, methods, headers

# ── Config ─────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2:1b"
DB_PATH = "./rag_db"
DOCS_DIR = "./docs"

# ── ChromaDB setup ─────────────────────────────────────────────────────────
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection("documents")


# ── Pydantic schemas ───────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    n_results: int = 3
    max_distance: float = 1.2

    # TODO: Add a @field_validator("question") that raises ValueError
    #   if the question is an empty string (after stripping whitespace)


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
# (Paste / adapt functions from your my_rag.py / rag_pipeline.py)

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer ONLY from the provided context. "
    "If the context doesn't contain the answer, say you don't have enough "
    "information. Cite source documents by name. Keep responses under 200 words."
)


def load_documents(directory: str) -> list[dict]:
    pass  # TODO: copy from previous exercises


def retrieve(query: str, n_results: int = 3, max_distance: float = 1.2) -> list[dict]:
    """Query ChromaDB, filter by max_distance, return chunk dicts."""
    pass  # TODO


def compute_confidence(chunks: list[dict]) -> str:
    pass  # TODO: "high" / "medium" / "low" based on best distance


def call_ollama(messages: list[dict]) -> str:
    """Call Ollama (non-streaming). Raise HTTPException 503 if unreachable."""
    # TODO: POST to Ollama
    # TODO: Catch ConnectionError → raise HTTPException(status_code=503, detail="...")
    pass  # TODO


def check_ollama_health() -> bool:
    """Return True if Ollama is reachable, False otherwise."""
    # TODO: GET {OLLAMA_URL}/api/tags — return True on 200, False on ConnectionError
    pass  # TODO


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Retrieve context and generate a grounded answer."""
    # TODO: retrieve() with req.n_results and req.max_distance
    # TODO: If no chunks, return AskResponse with a "no documents" message,
    #       empty sources, confidence "low"
    # TODO: Build messages (SYSTEM_PROMPT + context + question)
    # TODO: call_ollama() — may raise 503
    # TODO: Return AskResponse with answer, sources, confidence
    pass  # TODO


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """Load documents from docs/ into ChromaDB."""
    # TODO: load_documents(DOCS_DIR)
    # TODO: collection.upsert(...)
    # TODO: Return IngestResponse with chunks_ingested and a message
    pass  # TODO


@app.get("/stats")
def stats():
    """Return document count and model info."""
    # TODO: return {"document_count": collection.count(), "model": MODEL, "db_path": DB_PATH}
    pass  # TODO


@app.get("/health")
def health():
    """Check ChromaDB and Ollama liveness."""
    # TODO: Try collection.count() → chromadb status "ok" or "error"
    # TODO: check_ollama_health() → ollama status "connected" or "disconnected"
    # TODO: overall status "ok" only if both are healthy
    # TODO: return {"status": ..., "chromadb": ..., "ollama": ..., "document_count": ...}
    pass  # TODO
