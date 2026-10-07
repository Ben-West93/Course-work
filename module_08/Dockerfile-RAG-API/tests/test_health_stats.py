import pytest
import requests

import my_rag_api as api
from conftest import FakeResponse


class BrokenCollection:
    def count(self):
        raise RuntimeError("database is locked")


# ── /stats ─────────────────────────────────────────────────────────────────

def test_stats(client, ingested):
    r = client.get("/stats")
    assert r.status_code == 200
    assert r.json() == {"document_count": 32, "model": api.MODEL, "db_path": api.DB_PATH}


def test_stats_empty(client, empty_collection):
    assert client.get("/stats").json()["document_count"] == 0


def test_stats_when_chroma_is_broken(client, monkeypatch):
    monkeypatch.setattr(api, "collection", BrokenCollection())
    r = client.get("/stats")
    assert r.status_code == 503
    assert r.json() == {"detail": "ChromaDB is not readable"}


# ── /health ────────────────────────────────────────────────────────────────

def test_health_ok(client, ingested):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok", "chromadb": "ok", "ollama": "connected",
        "document_count": 32, "model": api.MODEL,
    }


@pytest.mark.parametrize("error, reply", [
    (requests.exceptions.ConnectionError(), None),
    (requests.exceptions.Timeout(), None),
    (requests.exceptions.InvalidURL(), None),
    (None, FakeResponse(500)),
    (None, FakeResponse(404)),
])
def test_health_degraded_without_ollama(client, ingested, ollama, error, reply):
    ollama.tags_error = error
    if reply:
        ollama.tags_reply = reply
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"
    assert r.json()["ollama"] == "disconnected"
    assert r.json()["chromadb"] == "ok"


@pytest.mark.parametrize("ollama_up", [True, False])
def test_health_error_when_chroma_is_broken(client, monkeypatch, ollama, ollama_up):
    monkeypatch.setattr(api, "collection", BrokenCollection())
    if not ollama_up:
        ollama.tags_error = requests.exceptions.ConnectionError()
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json() == {
        "status": "error", "chromadb": "error",
        "ollama": "connected" if ollama_up else "disconnected",
        "document_count": None, "model": api.MODEL,
    }


def test_health_empty_collection_is_still_ok(client, empty_collection):
    assert client.get("/health").json()["document_count"] == 0
    assert client.get("/health").json()["status"] == "ok"


def test_chroma_failure_is_logged(client, monkeypatch, caplog):
    monkeypatch.setattr(api, "collection", BrokenCollection())
    client.get("/stats")
    assert "database is locked" in caplog.text
