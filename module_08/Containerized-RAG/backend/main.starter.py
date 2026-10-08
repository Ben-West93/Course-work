"""
Module 8 Project — Containerized RAG Assistant  (STARTER)
Backend: FastAPI Application
=============================
Run locally:
    uvicorn main:app --reload --port 8000

Your task: implement the four endpoint handlers below.
The RAG logic lives in rag.py — import and call those functions here.

Required endpoints:
    GET  /          → welcome message
    POST /ask       → RAG query: retrieve + generate + return answer + sources
    POST /ingest    → load documents from ./docs and store in ChromaDB
    GET  /stats     → document count, model name, db path
    GET  /health    → ChromaDB status, Ollama connectivity, document count
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from config import settings
import rag

app = FastAPI(title="RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCS_DIR = "./docs"

# ── Request / Response schemas ─────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    n_results: int = 3
    max_distance: float = 1.2

    @field_validator("question")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
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


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    # TODO: Return a welcome dict, e.g. {"message": "RAG API is running", "docs": "/docs"}
    return {"message": "TODO: implement root endpoint"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    RAG query pipeline:
      1. rag.retrieve(req.question, req.n_results, req.max_distance)
      2. If no chunks, return a helpful "no documents" response
      3. rag.build_prompt(req.question, chunks)
      4. rag.generate(messages)
      5. rag.compute_confidence(chunks)
      6. Return AskResponse

    TODO: Implement this endpoint.
    """
    # TODO: implement
    return AskResponse(
        answer="TODO: implement the /ask endpoint",
        sources=[],
        confidence="low",
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """
    Load documents from DOCS_DIR and upsert into ChromaDB.
      1. rag.load_documents(DOCS_DIR)
      2. If empty, return a helpful message
      3. rag.collection.upsert(documents=..., metadatas=..., ids=...)
      4. Return IngestResponse with count

    TODO: Implement this endpoint.
    Hint: access the ChromaDB collection via rag.collection
    """
    # TODO: implement
    return IngestResponse(chunks_ingested=0, message="TODO: implement /ingest")


@app.get("/stats")
def stats():
    """
    Return document count, model name, and db path.
    TODO: Return a dict with "document_count", "model", and "db_path".
    Hint: rag.collection.count(), settings.model_name, settings.chroma_path
    """
    # TODO: implement
    return {"document_count": 0, "model": settings.model_name, "db_path": settings.chroma_path}


@app.get("/health")
def health():
    """
    Return status of ChromaDB and Ollama.
    TODO: Implement this endpoint.

    Suggested response shape:
        {
            "status":         "ok" | "degraded",
            "chromadb":       "ok" | "error",
            "ollama":         "connected" | "disconnected",
            "document_count": <int>,
        }

    To check Ollama: try GET settings.ollama_url + "/api/tags" with a short timeout.
    """
    # TODO: implement
    return {"status": "TODO", "chromadb": "unknown", "ollama": "unknown", "document_count": 0}
