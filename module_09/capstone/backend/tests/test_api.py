"""
API tests. No Postgres, Ollama or Docker needed:

    cd backend && pytest tests/ -v

SQLite and ChromaDB go to a throwaway folder and OLLAMA_URL points at a closed
port (conftest.py). Tests that need an answer patch rag.generate.

To run the same tests against Postgres (it behaves differently on NUL bytes,
integer overflow and cascades), point TEST_DATABASE_URL at an empty database:

    TEST_DATABASE_URL=postgresql+psycopg://jobfit:jobfit@postgres:5432/jobfit_test pytest tests/
"""

import os

import jwt
import pytest

import rag
from database import SessionLocal
from main import NO_MATCH_ANSWER, NO_POSTINGS_ANSWER
from models import JobPosting
from tests.conftest import signup

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


@pytest.fixture
def fake_answer(monkeypatch):
    calls = []

    def generate(messages):
        calls.append(messages)
        return "Example Logistics asks for Python, FastAPI and Docker [1].", False

    monkeypatch.setattr(rag, "generate", generate)
    return calls


# ── Health and auth ────────────────────────────────────────────────────────

def test_health_degraded_without_ollama(client):
    body = client.get("/health").json()
    assert body["database"] == "ok"
    assert body["chroma"] == "ok"
    assert body["ollama"] == "disconnected"
    assert body["status"] == "degraded"


def test_register_login_and_me(client):
    r = client.post("/auth/register", json={"username": "  Ben_W  ", "password": "longenough"})
    assert r.status_code == 201
    assert r.json()["username"] == "ben_w"
    assert "hashed_password" not in r.json()

    # usernames are case-insensitive
    assert client.post("/auth/register", json={"username": "BEN_W", "password": "longenough"}).status_code == 409

    token = client.post("/auth/token", data={"username": "Ben_W", "password": "longenough"}).json()
    assert token["token_type"] == "bearer"
    assert token["expires_in"] == 30 * 60
    me = client.get("/me", headers={"Authorization": f"Bearer {token['access_token']}"})
    assert me.json()["username"] == "ben_w"


def test_wrong_password_and_unknown_user_get_the_same_401(client):
    client.post("/auth/register", json={"username": "pw_check", "password": "the-password"})
    wrong = client.post("/auth/token", data={"username": "pw_check", "password": "not-the-password"})
    unknown = client.post("/auth/token", data={"username": "nobody_here", "password": "whatever1"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


@pytest.mark.parametrize("auth_header", [
    None,
    "Bearer not-a-jwt",
    # signed with a different secret
    "Bearer " + jwt.encode({"sub": "1", "exp": 4102444800}, "a-different-secret-of-at-least-32-bytes", algorithm="HS256"),
    # right secret, already expired
    "Bearer " + jwt.encode({"sub": "1", "exp": 1}, os.environ["JWT_SECRET"], algorithm="HS256"),
])
def test_protected_routes_reject_bad_tokens(client, auth_header):
    headers = {"Authorization": auth_header} if auth_header else {}
    for method, path in [("get", "/jobs"), ("post", "/ask"), ("get", "/me")]:
        r = client.request(method, path, headers=headers, json={"question": "anything?"})
        assert r.status_code == 401, (method, path)


# ── Postings ───────────────────────────────────────────────────────────────

def test_create_posting_indexes_its_chunks(client, headers):
    r = client.post("/jobs", headers=headers, json=POSTING)
    assert r.status_code == 201
    job = r.json()
    assert job["indexed"] is True
    stored = rag.collection.get(where={"job_id": job["id"]})
    assert stored["ids"]
    assert all(m["user_id"] == stored["metadatas"][0]["user_id"] for m in stored["metadatas"])
    # every chunk carries the header used for search
    header = "Junior Backend Developer at Example Logistics (Remote)"
    assert all(doc.startswith(header) for doc in stored["documents"])

    # the same text again, even with different spacing and case, is a duplicate
    again = dict(POSTING, text=POSTING["text"].upper().replace("\n\n", "\n \n"))
    assert client.post("/jobs", headers=headers, json=again).status_code == 409


def test_posting_validation(client, headers):
    assert client.post("/jobs", headers=headers, json=dict(POSTING, text="too short")).status_code == 422
    assert client.post("/jobs", headers=headers, json=dict(POSTING, title="   ")).status_code == 422
    assert client.post("/jobs", headers=headers, json=dict(POSTING, text="x" * 20_001)).status_code == 422


def test_users_cannot_see_each_others_postings(client, headers, fake_answer):
    job = client.post("/jobs", headers=headers, json=POSTING).json()
    other = signup(client)

    assert client.get(f"/jobs/{job['id']}", headers=other).status_code == 404
    assert client.patch(f"/jobs/{job['id']}", headers=other, json={"title": "Mine now"}).status_code == 404
    assert client.delete(f"/jobs/{job['id']}", headers=other).status_code == 404
    assert client.get("/jobs", headers=other).json() == []

    # the other user has no postings, so their question never reaches the model
    r = client.post("/ask", headers=other, json={"question": "Which job wants FastAPI?"})
    assert r.json()["answer"] == NO_POSTINGS_ANSWER
    assert fake_answer == []


def test_patch_reindexes_and_delete_removes_chunks(client, headers):
    job = client.post("/jobs", headers=headers, json=POSTING).json()

    r = client.patch(f"/jobs/{job['id']}", headers=headers, json={"company": "Renamed Freight"})
    assert r.status_code == 200
    docs = rag.collection.get(where={"job_id": job["id"]})["documents"]
    assert docs and all(" at Renamed Freight (Remote)" in d.split("\n", 1)[0] for d in docs)

    assert client.patch(f"/jobs/{job['id']}", headers=headers, json={"title": None}).status_code == 422

    assert client.delete(f"/jobs/{job['id']}", headers=headers).status_code == 204
    assert rag.collection.get(where={"job_id": job["id"]})["ids"] == []
    assert client.get(f"/jobs/{job['id']}", headers=headers).status_code == 404


def test_samples_load_once(client, headers):
    first = client.post("/jobs/samples", headers=headers).json()
    assert first["added"] == len(rag.load_samples()) > 0
    assert first["skipped"] == 0
    second = client.post("/jobs/samples", headers=headers).json()
    assert second == {"added": 0, "skipped": first["added"]}


# ── Ask ────────────────────────────────────────────────────────────────────

def test_ask_returns_grounded_answer_with_sources(client, headers, fake_answer):
    job = client.post("/jobs", headers=headers, json=POSTING).json()
    r = client.post("/ask", headers=headers, json={"question": "What does Example Logistics require?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].endswith("[1].")
    assert body["sources"] and body["sources"][0]["n"] == 1
    assert {s["job_id"] for s in body["sources"]} == {job["id"]}
    assert body["confidence"] in ("high", "medium")

    # the excerpts went into the prompt, numbered
    system, user = fake_answer[0]
    assert "using only the numbered excerpts" in system["content"]
    assert "[1] Junior Backend Developer at Example Logistics (Remote):" in user["content"]
    assert user["content"].endswith("Question: What does Example Logistics require?")


def test_ask_far_off_question_skips_the_model(client, headers, fake_answer, monkeypatch):
    client.post("/jobs", headers=headers, json=POSTING)
    # a cutoff nothing can pass, same as a question with no related chunk
    monkeypatch.setattr(rag.settings, "max_distance", 0.0)
    r = client.post("/ask", headers=headers, json={"question": "How do I bake sourdough bread?"})
    assert r.json()["answer"] == NO_MATCH_ANSWER
    assert r.json()["confidence"] == "low"
    assert fake_answer == []


def test_ask_when_ollama_is_down_is_503(client, headers):
    client.post("/jobs", headers=headers, json=POSTING)
    r = client.post("/ask", headers=headers, json={"question": "What does Example Logistics require?"})
    assert r.status_code == 503
    assert r.json()["detail"] == "Ollama unavailable"


def test_ask_rejects_someone_elses_job_ids(client, headers):
    job = client.post("/jobs", headers=headers, json=POSTING).json()
    other = signup(client)
    client.post("/jobs/samples", headers=other)
    r = client.post("/ask", headers=other, json={"question": "What is required?", "job_ids": [job["id"]]})
    assert r.status_code == 404


# ── Pipeline pieces ────────────────────────────────────────────────────────

def test_chunks_stay_under_the_limit_and_keep_headings():
    for sample in rag.load_samples():
        chunks = rag.chunk_posting(sample["text"])
        assert all(len(c) <= rag.MAX_CHUNK_CHARS for c in chunks)
        # a heading is never the last line of a chunk, cut off from its section
        assert not any(c.splitlines()[-1].startswith("#") for c in chunks)


def test_named_company_limits_the_search():
    class Job:
        def __init__(self, id, company):
            self.id, self.company = id, company

    jobs = [Job(1, "Harbor & Pine Insurance"), Job(2, "Cobalt Freight"), Job(3, "AI Labs")]
    assert rag.named_postings("What does Harbor & Pine want for SQL?", jobs) == [1]
    assert rag.named_postings("what does cobalt want?", jobs) == [2]
    assert rag.named_postings("Which jobs ask for Docker?", jobs) is None
    # "ai" is too short to count as naming AI Labs
    assert rag.named_postings("Which AI jobs are remote?", jobs) is None


@pytest.mark.parametrize("answer, expected", [
    ("Python is required [1].", "high"),
    ("Python is required.", "low"),
    ("Python is required [7].", "low"),
    ("The saved postings don't have enough information to answer that.", "low"),
])
def test_confidence(answer, expected):
    chunks = [{"distance": 0.5}, {"distance": 0.9}]
    assert rag.compute_confidence(chunks, answer) == expected


def test_deleting_account_cascades(client):
    headers = signup(client)
    job = client.post("/jobs", headers=headers, json=POSTING).json()
    assert client.delete("/me", headers=headers).status_code == 204
    with SessionLocal() as db:
        assert db.get(JobPosting, job["id"]) is None
    assert rag.collection.get(where={"job_id": job["id"]})["ids"] == []
    # the token is still signed and unexpired, but the user is gone
    assert client.get("/me", headers=headers).status_code == 401
