"""
API and pipeline tests. No Ollama or Docker needed:

    cd backend && pytest tests/ -v

ChromaDB goes to a throwaway folder and OLLAMA_URL points at a closed port, so
the tests never touch the real db or a running model. Where a test needs Ollama
to answer (or fail in a specific way) it patches requests.post / rag.generate.
"""

import atexit
import os
import shutil
import tempfile

# has to happen before config.py is imported, settings are read once at import
TEST_DB = tempfile.mkdtemp(prefix="rag_test_")
atexit.register(shutil.rmtree, TEST_DB, ignore_errors=True)
os.environ["CHROMA_PATH"] = TEST_DB
os.environ["OLLAMA_URL"] = "http://127.0.0.1:9"

import pytest
import requests
from fastapi.testclient import TestClient

import rag
from main import NO_DOCUMENTS_ANSWER, NO_MATCH_ANSWER, app

client = TestClient(app)


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(str(self.status_code))


def clear_collection():
    ids = rag.collection.get(include=[])["ids"]
    if ids:
        rag.collection.delete(ids=ids)


@pytest.fixture
def empty_db():
    clear_collection()
    yield
    clear_collection()


@pytest.fixture
def ingested():
    clear_collection()
    r = client.post("/ingest")
    assert r.status_code == 200
    yield r.json()
    clear_collection()


@pytest.fixture
def fake_answer(monkeypatch):
    monkeypatch.setattr(rag, "generate", lambda messages: "List comprehensions build lists.")


# ── Required smoke tests ───────────────────────────────────────────────────

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "message" in body
    assert body["docs"] == "/docs"


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    for key in ("status", "chromadb", "ollama", "document_count"):
        assert key in body
    assert body["chromadb"] == "ok"
    assert isinstance(body["document_count"], int)


def test_stats():
    r = client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["document_count"], int)
    assert body["model"]
    assert body["db_path"] == os.path.normpath(TEST_DB)


# ── /health states ─────────────────────────────────────────────────────────

def test_health_degraded_when_ollama_down():
    # OLLAMA_URL points at a closed port
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["ollama"] == "disconnected"
    assert body["model_pulled"] is False


def test_health_ok_when_everything_is_up(monkeypatch):
    monkeypatch.setattr(rag, "ollama_status", lambda: (True, True))
    assert client.get("/health").json()["status"] == "ok"


def test_health_degraded_when_model_not_pulled(monkeypatch):
    tags = FakeResponse(200, {"models": [{"name": "some-other-model:latest"}]})
    monkeypatch.setattr(rag.requests, "get", lambda *a, **kw: tags)
    body = client.get("/health").json()
    assert (body["status"], body["ollama"], body["model_pulled"]) == ("degraded", "connected", False)


class BrokenCollection:
    def __getattr__(self, name):
        def fail(*args, **kwargs):
            raise RuntimeError("database disk image is malformed")
        return fail


def test_chromadb_failure_is_503_not_500(monkeypatch):
    monkeypatch.setattr(rag, "collection", BrokenCollection())

    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "error"
    assert r.json()["chromadb"] == "error"
    assert r.json()["document_count"] == 0

    assert client.get("/stats").status_code == 503
    assert client.post("/ask", json={"question": "What is Python?"}).json() == {"detail": "ChromaDB search failed"}
    assert client.post("/ingest").json() == {"detail": "Could not write to ChromaDB"}


# ── /ingest ────────────────────────────────────────────────────────────────

def test_ingest_counts(ingested):
    assert ingested["chunks_ingested"] > 0
    assert ingested["files_ingested"] == 5
    assert ingested["chunks_removed"] == 0
    assert client.get("/stats").json()["document_count"] == ingested["chunks_ingested"]


def test_ingest_twice_does_not_duplicate(ingested):
    again = client.post("/ingest").json()
    assert again["chunks_ingested"] == ingested["chunks_ingested"]
    assert rag.collection.count() == ingested["chunks_ingested"]


def test_ingest_removes_stale_chunks(ingested):
    rag.collection.add(documents=["left over from a deleted file"], ids=["deleted.txt_0"],
                       metadatas=[{"source": "deleted.txt", "chunk_index": 0}])
    body = client.post("/ingest").json()
    assert body["chunks_removed"] == 1
    assert "removed 1 stale" in body["message"]
    assert rag.collection.get(ids=["deleted.txt_0"])["ids"] == []


def test_ingest_with_no_docs_leaves_db_alone(ingested, monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "DOCS_DIR", str(tmp_path))
    body = client.post("/ingest").json()
    assert body["chunks_ingested"] == 0
    assert "nothing changed" in body["message"]
    assert rag.collection.count() == ingested["chunks_ingested"]


# ── /ask: valid requests ───────────────────────────────────────────────────

def test_ask_answers_with_sources(ingested, fake_answer):
    r = client.post("/ask", json={"question": "What is a list comprehension?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered"
    assert body["answer"]
    assert body["confidence"] in ("high", "medium", "low")
    assert body["sources"]
    assert body["sources"][0]["source"] == "python_basics.txt"
    for s in body["sources"]:
        assert set(s) == {"text", "source", "distance"}
        assert s["text"].strip()
        assert s["distance"] <= 1.2
    distances = [s["distance"] for s in body["sources"]]
    assert distances == sorted(distances)


def test_ask_empty_db_falls_back(empty_db, monkeypatch):
    monkeypatch.setattr(rag, "generate", lambda m: pytest.fail("Ollama shouldn't be called"))
    body = client.post("/ask", json={"question": "What is Python?"}).json()
    assert body == {"answer": NO_DOCUMENTS_ANSWER, "sources": [], "confidence": "low", "status": "no_documents"}


def test_ask_off_topic_falls_back(ingested, monkeypatch):
    monkeypatch.setattr(rag, "generate", lambda m: pytest.fail("Ollama shouldn't be called"))
    body = client.post("/ask", json={"question": "Who won the 2018 World Cup?"}).json()
    assert body == {"answer": NO_MATCH_ANSWER, "sources": [], "confidence": "low", "status": "no_match"}


def test_ask_strips_sources_past_max_distance(ingested, fake_answer):
    loose = client.post("/ask", json={"question": "What is a list comprehension?", "n_results": 10,
                                      "max_distance": 4}).json()
    strict = client.post("/ask", json={"question": "What is a list comprehension?", "n_results": 10,
                                       "max_distance": 0.7}).json()
    assert len(strict["sources"]) < len(loose["sources"])
    assert all(s["distance"] <= 0.7 for s in strict["sources"])


def test_ask_borderline_question_is_low(ingested, fake_answer):
    # top chunk lands between CONFIDENCE_THRESHOLD (1.0) and MAX_DISTANCE (1.2)
    body = client.post("/ask", json={"question": "What is Pydantic?"}).json()
    assert body["status"] == "answered"
    assert 1.0 < body["sources"][0]["distance"] <= 1.2
    assert body["confidence"] == "low"


def test_ask_strips_whitespace_from_question(ingested, monkeypatch):
    sent = []

    def fake_generate(messages):
        sent.append(messages[-1]["content"])
        return "ok"

    monkeypatch.setattr(rag, "generate", fake_generate)
    client.post("/ask", json={"question": "   What is a list comprehension?  \n"})
    assert sent == ["What is a list comprehension?"]


# ── /ask: invalid requests -> 422 ──────────────────────────────────────────

@pytest.mark.parametrize("body", [
    {"question": ""},
    {"question": "   \n\t "},
    {"question": "\u200b\ufeff"},
    {"question": "\x00"},
    {"question": None},
    {"question": 42},
    {"question": ["what", "is", "python"]},
    {},
    {"question": "x" * 1001},
    {"question": "ok", "n_results": 0},
    {"question": "ok", "n_results": -3},
    {"question": "ok", "n_results": 21},
    {"question": "ok", "n_results": "5"},
    {"question": "ok", "n_results": 2.5},
    {"question": "ok", "n_results": True},
    {"question": "ok", "max_distance": -0.1},
    {"question": "ok", "max_distance": 4.5},
    {"question": "ok", "max_distance": "1.0"},
])
def test_ask_rejects_bad_input(body):
    r = client.post("/ask", json=body)
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], list)


@pytest.mark.parametrize("raw", [
    b"not json",
    b'["a list", "not an object"]',
    b'{"question": "ok", "max_distance": NaN}',
    b'{"question": "ok", "max_distance": Infinity}',
    b'{"question": "\\ud800"}',
])
def test_ask_rejects_unparseable_bodies_without_500(raw):
    r = client.post("/ask", content=raw, headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_empty_question_message():
    detail = client.post("/ask", json={"question": "  "}).json()["detail"]
    assert "question must not be empty" in detail[0]["msg"]


# ── /ask: Ollama failures -> 503 ───────────────────────────────────────────

def raise_(exc):
    def fake(*args, **kwargs):
        raise exc
    return fake


@pytest.mark.parametrize("fake_post, detail", [
    (raise_(requests.exceptions.ConnectionError("refused")), "Ollama unavailable"),
    (raise_(requests.exceptions.ReadTimeout("slow")), "Ollama timed out"),
    (lambda *a, **kw: FakeResponse(404, text='{"error":"model not found"}'),
     f"Ollama returned 404 for model {rag.settings.model_name}"),
    (lambda *a, **kw: FakeResponse(200, None, "<html>"), "Ollama's response had no answer in it"),
    (lambda *a, **kw: FakeResponse(200, {"message": {"content": "   "}}), "Ollama returned an empty answer"),
])
def test_ask_ollama_failures_are_503(ingested, monkeypatch, fake_post, detail):
    monkeypatch.setattr(rag.requests, "post", fake_post)
    r = client.post("/ask", json={"question": "What is a list comprehension?"})
    assert r.status_code == 503
    assert r.json() == {"detail": detail}


def test_ask_with_real_closed_port_is_503(ingested):
    # no patching, OLLAMA_URL really points at a closed port
    r = client.post("/ask", json={"question": "What is a list comprehension?"})
    assert r.status_code == 503
    assert r.json()["detail"] == "Ollama unavailable"


# ── confidence scoring ─────────────────────────────────────────────────────

def chunk(distance, source="a.txt"):
    return {"text": "t", "source": source, "distance": distance}


@pytest.mark.parametrize("chunks, expected", [
    ([], "low"),
    ([chunk(0.3)], "high"),
    ([chunk(0.4999)], "high"),
    ([chunk(0.5)], "medium"),
    ([chunk(0.95)], "medium"),
    ([chunk(1.0)], "medium"),
    ([chunk(1.0001)], "low"),
    ([chunk(1.15)], "low"),
    # nearly tied files at the top: ambiguous
    ([chunk(0.40, "a.txt"), chunk(0.43, "b.txt")], "low"),
    ([chunk(0.70, "a.txt"), chunk(0.74, "b.txt")], "low"),
    # same gap but the same file is just more of one topic
    ([chunk(0.40, "a.txt"), chunk(0.43, "a.txt")], "high"),
    # clear winner
    ([chunk(0.40, "a.txt"), chunk(0.60, "b.txt")], "high"),
    # order doesn't matter
    ([chunk(0.60, "b.txt"), chunk(0.40, "a.txt")], "high"),
])
def test_compute_confidence(chunks, expected):
    assert rag.compute_confidence(chunks) == expected


@pytest.mark.parametrize("answer", [
    "I don't have enough information to answer that.",
    "The context does not contain enough information about this.",
    "There is no enough information in the sources.",
])
def test_refusal_is_low_even_with_close_match(answer):
    assert rag.compute_confidence([chunk(0.2)], answer) == "low"


def test_normal_answer_keeps_confidence():
    assert rag.compute_confidence([chunk(0.2, "python_basics.txt")], "Use one [python_basics.txt].") == "high"


def test_made_up_citation_is_low():
    chunks = [chunk(0.2, "python_basics.txt")]
    assert rag.compute_confidence(chunks, "See python_tutorial.md for details.") == "low"
    # a real corpus file that wasn't retrieved for this question counts too
    assert rag.compute_confidence(chunks, "Source: docker_basics.txt") == "low"


def test_file_named_inside_a_chunk_is_not_a_made_up_citation():
    chunks = [{"text": "COPY requirements.txt . before the code", "source": "docker_basics.txt", "distance": 0.3}]
    assert rag.compute_confidence(chunks, "Copy requirements.txt first [docker_basics.txt].") == "high"


# ── chunking and retrieval ─────────────────────────────────────────────────

def test_heading_is_merged_into_next_paragraph():
    chunks = rag.chunk_text("Python Functions\n\nFunctions are defined with def and can take arguments.")
    assert chunks == ["Python Functions\n\nFunctions are defined with def and can take arguments."]


def test_code_block_stays_with_its_lead_in():
    text = ("Creating an app\n\nA minimal app needs only a few lines:\n\n    app = FastAPI()\n\n"
            "    @app.get('/')\n    def root(): ...")
    chunks = rag.chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].startswith("Creating an app") and chunks[0].endswith("def root(): ...")


def test_short_tail_joins_previous_paragraph():
    text = "Any parameter that is not a path parameter is treated as a query parameter by FastAPI.\n\nLike: GET /x?q=1"
    assert len(rag.chunk_text(text)) == 1


def test_long_paragraph_is_split():
    text = " ".join(["This sentence is about forty characters."] * 80)
    chunks = rag.chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= rag.MAX_CHUNK_CHARS for c in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_crlf_and_blank_files(tmp_path):
    (tmp_path / "windows.txt").write_bytes(
        b"Windows Notes\r\n\r\nThis file was saved with CRLF line endings on purpose for the test.\r\n")
    (tmp_path / "empty.md").write_text("\n\n   \n")
    (tmp_path / ".hidden.txt").write_text("Hidden files are skipped even when they end in .txt for sure.")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "bad_bytes.txt").write_bytes(b"Caf\xe9 au lait is fine even though this byte is not valid UTF-8.")
    (tmp_path / "subdir.txt").mkdir()

    docs = rag.load_documents(str(tmp_path))
    assert sorted(d["metadata"]["source"] for d in docs) == ["bad_bytes.txt", "windows.txt"]
    windows = next(d for d in docs if d["metadata"]["source"] == "windows.txt")
    assert "\r" not in windows["text"]
    assert windows["id"] == "windows.txt_0"
    assert windows["metadata"]["chunk_index"] == 0


def test_missing_docs_dir():
    assert rag.load_documents("/definitely/not/here") == []


def test_odd_metadata_and_duplicates_are_cleaned(empty_db):
    text = "Ollama runs large language models locally and serves them over a REST API."
    rag.collection.add(
        documents=[text, text, "Ollama listens on port 11434 by default for API requests."],
        metadatas=[{"source": 5}, {"source": "dupe.txt"}, {"chunk_index": 0}],
        ids=["x_0", "x_1", "x_2"],
    )
    chunks = rag.retrieve("What is Ollama?", n_results=10, max_distance=4)
    # the duplicate text only fills one source slot
    assert len(chunks) == 2
    assert [c["text"] for c in chunks].count(text) == 1
    # a number or a missing key never comes back as a source name
    port = next(c for c in chunks if "11434" in c["text"])
    assert port["source"] == "unknown"
    assert all(c["source"] in ("unknown", "dupe.txt") for c in chunks)


def test_n_results_larger_than_collection(empty_db):
    rag.collection.add(documents=["only one chunk stored here"], ids=["one_0"], metadatas=[{"source": "one.txt"}])
    assert len(rag.retrieve("chunk", n_results=20, max_distance=4)) == 1
