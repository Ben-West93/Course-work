import json
import os
import tempfile
import uuid

import pytest

# config reads the env once, on first import, and my_rag_api opens its db at
# import time, so all of this has to be set first. load_dotenv never overrides,
# so whatever is in a local .env (DEBUG=true, another model...) can't leak in
os.environ.update({
    "CHROMA_PATH": tempfile.mkdtemp(prefix="rag_test_db_"),
    "OLLAMA_URL": "http://localhost:11434",
    "MODEL_NAME": "llama3.2:1b",
    "MAX_RESULTS": "3",
    "CONFIDENCE_THRESHOLD": "1.0",
    "DEBUG": "false",
})

from fastapi.testclient import TestClient  # noqa: E402

import my_rag_api as api  # noqa: E402


class FakeResponse:
    def __init__(self, status_code=200, data=None, bad_json=False):
        self.status_code = status_code
        self._data = data
        self._bad_json = bad_json
        self.text = "<html>Bad Gateway</html>" if bad_json else json.dumps(data)

    def json(self):
        if self._bad_json:
            raise json.JSONDecodeError("Expecting value", "<html>", 0)
        return self._data


# stands in for requests.post/get so no test ever talks to a real Ollama
class FakeOllama:
    def __init__(self):
        self.answer = "Answer based on the context."
        self.chat_reply = None
        self.chat_error = None
        self.tags_reply = FakeResponse(200, {"models": [{"name": api.settings.model_name}]})
        self.tags_error = None
        self.calls = []

    def reply(self, status_code=200, data=None, bad_json=False):
        self.chat_reply = FakeResponse(status_code, data, bad_json)

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if self.chat_error:
            raise self.chat_error
        if self.chat_reply:
            return self.chat_reply
        return FakeResponse(200, {"message": {"role": "assistant", "content": self.answer}})

    def get(self, url, timeout=None):
        if self.tags_error:
            raise self.tags_error
        return self.tags_reply


class BrokenCollection:
    def __init__(self, error="database is locked"):
        self.error = error

    def _fail(self, *args, **kwargs):
        raise RuntimeError(self.error)

    count = query = upsert = get = delete = _fail


# returns exactly the results it's given, for distances and metadata a real
# collection would be awkward to produce
class ScriptedCollection:
    def __init__(self, rows):
        self.rows = rows  # (document, metadata, distance), closest first

    def count(self):
        return len(self.rows)

    def query(self, query_texts, n_results):
        rows = self.rows[:n_results]
        return {
            "documents": [[r[0] for r in rows]],
            "metadatas": [[r[1] for r in rows]],
            "distances": [[r[2] for r in rows]],
        }


@pytest.fixture
def scripted(monkeypatch):
    def use(*rows):
        monkeypatch.setattr(api, "collection", ScriptedCollection(list(rows)))
    return use


@pytest.fixture(autouse=True)
def ollama(monkeypatch):
    fake = FakeOllama()
    monkeypatch.setattr(api.requests, "post", fake.post)
    monkeypatch.setattr(api.requests, "get", fake.get)
    return fake


@pytest.fixture
def client():
    return TestClient(api.app)


@pytest.fixture
def empty_collection(monkeypatch):
    col = api.client.create_collection(f"test_{uuid.uuid4().hex}")
    monkeypatch.setattr(api, "collection", col)
    yield col
    api.client.delete_collection(col.name)


@pytest.fixture(scope="session")
def docs_collection():
    # embedding the real docs is the slow part, so do it once and share it
    # between the tests that only read
    col = api.client.create_collection("test_docs")
    chunks = api.load_documents(api.DOCS_DIR)
    col.upsert(
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        ids=[c["id"] for c in chunks],
    )
    return col


@pytest.fixture
def ingested(monkeypatch, docs_collection):
    monkeypatch.setattr(api, "collection", docs_collection)
    return docs_collection


# empty docs folder for /ingest to read, tests write their own files into it
@pytest.fixture
def docs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DOCS_DIR", str(tmp_path))
    return tmp_path
