"""
Shared setup for every backend test file: test_auth.py, test_crud.py and
test_rag.py (one per part of the API), test_edge_cases.py and test_pipeline.py.

By default the suite needs nothing running: SQLite and ChromaDB go to a
throwaway temp folder, OLLAMA_URL points at a closed port (so any test that
forgets to mock Ollama fails with a 503 rather than calling a real model),
and the JWT secret is fixed so tests can sign their own tokens.

TEST_DATABASE_URL runs the same tests against Postgres instead. The suite
DROPS EVERY TABLE in that database before it starts, so it refuses to run
unless the database name contains "test". Never point it at `jobfit`.

The environment has to be set before config.py is imported, because settings
are read once at import time. That's why the imports below come after it.
"""

import atexit
import os
import shutil
import tempfile

TEST_DIR = tempfile.mkdtemp(prefix="jobfit_test_")
atexit.register(shutil.rmtree, TEST_DIR, ignore_errors=True)

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")
if TEST_DATABASE_URL and "test" not in TEST_DATABASE_URL.rsplit("/", 1)[-1]:
    raise SystemExit(f"TEST_DATABASE_URL must name a test database (e.g. .../jobfit_test), "
                     f"the suite drops all its tables. Got {TEST_DATABASE_URL.rsplit('@', 1)[-1]!r}")

os.environ["DATABASE_URL"] = TEST_DATABASE_URL or f"sqlite:///{os.path.join(TEST_DIR, 'test.db')}"
os.environ["CHROMA_PATH"] = os.path.join(TEST_DIR, "chroma")
os.environ["OLLAMA_URL"] = "http://127.0.0.1:9"
os.environ["JWT_SECRET_KEY"] = "test-secret-only-for-the-pytest-suite-0123456789"

import uuid

import pytest
from fastapi.testclient import TestClient

import database
import rag_pipeline
from main import app

# a realistic posting for test_auth.py, test_crud.py and test_rag.py
POSTING = {
    "title": "Junior Backend Developer",
    "company": "Example Logistics",
    "location": "Remote",
    "text": (
        "Example Logistics moves freight for regional shippers.\n\n"
        "## Requirements\n\n"
        "- 2+ years of Python and FastAPI\n"
        "- Strong SQL and PostgreSQL experience\n"
        "- Docker for local development and deployment\n\n"
        "## Nice to have\n\n"
        "- Redis for caching and job queues\n"
        "- AWS experience (ECS, RDS)\n"
    ),
}


@pytest.fixture(scope="session")
def client():
    # a reused Postgres test database starts empty every run
    import models  # noqa: F401

    database.Base.metadata.drop_all(database.engine)
    # the context manager runs the lifespan, which creates the tables
    with TestClient(app) as c:
        yield c


def signup(client) -> dict:
    """A fresh user's auth headers."""
    username = f"user_{uuid.uuid4().hex[:8]}"
    r = client.post("/auth/register", json={"username": username, "password": "correct-horse"})
    assert r.status_code == 201
    r = client.post("/auth/token", data={"username": username, "password": "correct-horse"})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
def headers(client):
    return signup(client)


@pytest.fixture
def fake_answer(monkeypatch):
    """Stands in for the model. Returns the list of prompts it was sent."""
    calls = []

    def generate_answer(messages):
        calls.append(messages)
        return "Example Logistics asks for Python, FastAPI and Docker [1].", False

    monkeypatch.setattr(rag_pipeline, "generate_answer", generate_answer)
    return calls
