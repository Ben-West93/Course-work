"""
Shared setup for test_api.py and test_edge_cases.py.

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
os.environ["JWT_SECRET"] = "test-secret-only-for-the-pytest-suite-0123456789"

import uuid

import pytest
from fastapi.testclient import TestClient

import database
from main import app


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
