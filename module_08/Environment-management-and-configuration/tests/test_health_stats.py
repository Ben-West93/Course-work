import os

import pytest
import requests

import my_rag_api as api
from conftest import BrokenCollection, FakeResponse


def tags(*models):
    return FakeResponse(200, {"models": list(models)})


# ── /stats ─────────────────────────────────────────────────────────────────

def test_stats(client, ingested):
    r = client.get("/stats")
    assert r.status_code == 200
    assert r.json() == {
        "document_count": 32, "model": "llama3.2:1b", "db_path": api.DB_PATH,
        "ollama_url": "http://localhost:11434", "max_results": 3,
        "confidence_threshold": 1.0, "debug": False,
    }


def test_db_path_comes_from_the_env_var_compose_sets(client, empty_collection):
    # conftest sets CHROMA_PATH before the import, the same way compose does
    assert api.DB_PATH == os.environ["CHROMA_PATH"]
    assert client.get("/stats").json()["db_path"] == os.environ["CHROMA_PATH"]


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
        "document_count": 32, "model": api.settings.model_name, "model_pulled": True,
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
    # can't tell, rather than claiming it isn't pulled
    assert r.json()["model_pulled"] is None


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
        "document_count": None, "model": api.settings.model_name,
        "model_pulled": True if ollama_up else None,
    }


def test_health_empty_collection_is_still_ok(client, empty_collection):
    assert client.get("/health").json()["document_count"] == 0
    assert client.get("/health").json()["status"] == "ok"


def test_chroma_failure_is_logged(client, monkeypatch, caplog):
    monkeypatch.setattr(api, "collection", BrokenCollection())
    client.get("/stats")
    assert "database is locked" in caplog.text


# ── /health: is MODEL_NAME actually pulled ─────────────────────────────────

@pytest.mark.parametrize("model, listed", [
    ("llama3.2:1b", [{"name": "llama3.2:1b", "model": "llama3.2:1b"}]),
    ("llama3.2:1b", [{"name": "qwen2.5:0.5b"}, {"name": "llama3.2:1b"}]),
    ("llama3.2:1b", [{"model": "llama3.2:1b"}]),
    # no tag means :latest, on either side
    ("llama3.2", [{"name": "llama3.2:latest"}]),
    ("llama3.2:latest", [{"name": "llama3.2"}]),
    ("myregistry:5000/team/model", [{"name": "myregistry:5000/team/model:latest"}]),
])
def test_health_ok_when_the_model_is_pulled(client, ingested, ollama, monkeypatch, model, listed):
    monkeypatch.setattr(api.settings, "model_name", model)
    ollama.tags_reply = tags(*listed)
    body = client.get("/health").json()
    assert (body["status"], body["model"], body["model_pulled"]) == ("ok", model, True)


@pytest.mark.parametrize("model, listed", [
    ("llama3.2:3b", [{"name": "llama3.2:1b"}]),             # changed in .env, never pulled
    ("llama3.2", [{"name": "llama3.2:1b"}]),                # means llama3.2:latest
    ("llama3.2:1B", [{"name": "llama3.2:1b"}]),             # ollama names are case sensitive
    ("llama3.2:1b", []),                                    # nothing pulled yet
])
def test_health_degraded_when_the_model_isnt_pulled(client, ingested, ollama, monkeypatch, model, listed):
    monkeypatch.setattr(api.settings, "model_name", model)
    ollama.tags_reply = tags(*listed)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "degraded", "chromadb": "ok", "ollama": "connected",
        "document_count": 32, "model": model, "model_pulled": False,
    }


@pytest.mark.parametrize("reply", [
    FakeResponse(200, bad_json=True),
    FakeResponse(200, {}),
    FakeResponse(200, {"models": None}),
    FakeResponse(200, {"models": [None]}),
    FakeResponse(200, {"models": [{"name": None}]}),
    FakeResponse(200, {"models": [{"name": 5}]}),
    FakeResponse(200, ["llama3.2:1b"]),
])
def test_health_unreadable_model_list(client, ingested, ollama, caplog, reply):
    # something answered 200 on the port, it just isn't a usable model list
    ollama.tags_reply = reply
    body = client.get("/health").json()
    assert (body["status"], body["ollama"], body["model_pulled"]) == ("degraded", "connected", False)
    assert "Couldn't read the model list" in caplog.text


def test_health_model_check_doesnt_matter_when_chroma_is_broken(client, monkeypatch, ollama):
    monkeypatch.setattr(api, "collection", BrokenCollection())
    ollama.tags_reply = tags()
    r = client.get("/health")
    assert r.status_code == 503
    assert (r.json()["status"], r.json()["model_pulled"]) == ("error", False)


# ── settings reach every endpoint ──────────────────────────────────────────
# the endpoints read settings when they run, so changing the shared object is
# the same as starting the app with a different .env

def test_changed_settings_show_up_without_code_changes(client, ingested, ollama, monkeypatch):
    monkeypatch.setattr(api.settings, "model_name", "llama3.2:3b")
    monkeypatch.setattr(api.settings, "ollama_url", "http://ollama:11434")
    monkeypatch.setattr(api.settings, "debug", True)

    stats = client.get("/stats").json()
    assert (stats["model"], stats["ollama_url"], stats["debug"]) == ("llama3.2:3b", "http://ollama:11434", True)
    assert client.get("/health").json()["model"] == "llama3.2:3b"

    client.post("/ask", json={"question": "What is chunking?"})
    assert ollama.calls[0]["url"] == "http://ollama:11434/api/chat"
    assert ollama.calls[0]["json"]["model"] == "llama3.2:3b"


def test_stats_types_match_the_settings(client, empty_collection):
    stats = client.get("/stats").json()
    assert type(stats["max_results"]) is int
    assert type(stats["confidence_threshold"]) is float
    assert type(stats["debug"]) is bool


# ── CHROMA_PATH ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path, expected", [
    ("./rag_db", os.path.join(api.BASE_DIR, "rag_db")),
    ("rag_db/", os.path.join(api.BASE_DIR, "rag_db")),
    ("data/../rag_db", os.path.join(api.BASE_DIR, "rag_db")),
    ("/app/rag_db", "/app/rag_db"),
    ("/app/rag_db/", "/app/rag_db"),
    ("~/rag_db", os.path.join(os.path.expanduser("~"), "rag_db")),
])
def test_chroma_dir(path, expected):
    assert api.chroma_dir(path) == expected
