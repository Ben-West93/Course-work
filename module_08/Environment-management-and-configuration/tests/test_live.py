# End-to-end checks against the running compose stack with the real model.
# Skipped unless RAG_API_URL is set:
#   docker-compose up -d
#   docker-compose exec ollama ollama pull llama3.2:1b
#   RAG_API_URL=http://localhost:8000 pytest tests/test_live.py

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx2 as httpx
import pytest
from dotenv import dotenv_values

BASE = os.getenv("RAG_API_URL")
pytestmark = pytest.mark.skipif(not BASE, reason="RAG_API_URL not set")


@pytest.fixture(scope="module")
def http():
    with httpx.Client(base_url=BASE or "", timeout=180) as c:
        yield c


@pytest.fixture(scope="module", autouse=True)
def ingested(http):
    r = http.post("/ingest")
    assert r.status_code == 200
    return r.json()


def test_docs(http):
    assert http.get("/docs").status_code == 200


def test_root(http):
    assert http.get("/").json() == {"message": "RAG API is running", "docs": "/docs"}


def test_health_ok(http):
    body = http.get("/health").json()
    assert body["status"] == "ok", "is the ollama service up? (docker-compose ps)"
    assert body["document_count"] == 32


def test_ingest(ingested):
    assert (ingested["chunks_ingested"], ingested["files_ingested"]) == (32, 7)


def test_stats(http):
    assert http.get("/stats").json()["document_count"] == 32


def test_stats_shows_the_settings_from_env_file(http):
    # the container's settings come from .env, apart from the two compose fixes
    env = dotenv_values(Path(__file__).resolve().parent.parent / ".env")
    stats = http.get("/stats").json()
    assert stats["model"] == env.get("MODEL_NAME", "llama3.2:1b")
    assert stats["max_results"] == int(env.get("MAX_RESULTS", "3"))
    assert stats["confidence_threshold"] == float(env.get("CONFIDENCE_THRESHOLD", "1.0"))
    assert stats["ollama_url"] == "http://ollama:11434"
    assert stats["db_path"] == "/app/rag_db"
    assert http.get("/health").json()["model"] == stats["model"]


@pytest.mark.parametrize("question, source", [
    ("How does ChromaDB store embeddings?", "chromadb.md"),
    ("What is prompt injection?", "prompt_injection.md"),
    ("How do I pull a model with Ollama?", "ollama.txt"),
])
def test_ask_real_model(http, question, source):
    r = http.post("/ask", json={"question": question})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].strip()
    assert body["sources"][0]["source"] == source
    assert body["confidence"] in {"high", "medium", "low"}


@pytest.mark.parametrize("question", ["What license is ChromaDB released under?", "Who created Ollama?"])
def test_not_in_the_docs_is_low(http, question):
    # both retrieve something (0.82 and 1.14) but the docs don't answer them.
    # max_distance is set so the result doesn't depend on CONFIDENCE_THRESHOLD
    body = http.post("/ask", json={"question": question, "max_distance": 1.2}).json()
    assert body["sources"]
    assert body["confidence"] == "low"


def test_ask_off_topic(http):
    body = http.post("/ask", json={"question": "Give me a pancake recipe"}).json()
    assert body["sources"] == []
    assert body["confidence"] == "low"


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "hi", "n_results": 0}, {}])
def test_ask_validation(http, payload):
    assert http.post("/ask", json=payload).status_code == 422


def test_concurrent_requests(http):
    with ThreadPoolExecutor(max_workers=10) as pool:
        health = list(pool.map(lambda _: http.get("/health").status_code, range(30)))
        asks = list(pool.map(
            lambda q: http.post("/ask", json={"question": q}).status_code,
            ["What is chunking?", "What is ChromaDB?", "What are embeddings?"],
        ))
    assert health == [200] * 30
    assert asks == [200] * 3
