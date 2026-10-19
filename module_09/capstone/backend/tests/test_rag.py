"""
RAG tests: /ask returns an answer with its sources, handles the "nothing
relevant saved" case without calling the model, and /health reports the RAG
services. Ollama is never called: tests that need an answer use the
fake_answer fixture (conftest.py), the rest find OLLAMA_URL closed.

    cd backend && pytest tests/test_rag.py -v

test_pipeline.py goes through each stage of the pipeline in more detail.
"""

import pytest

import rag_pipeline
from main import NO_MATCH_ANSWER
from tests.conftest import POSTING, signup


def test_health_returns_200(client, monkeypatch):
    # Ollama connected with the model pulled, the one service the suite has no real copy of
    monkeypatch.setattr(rag_pipeline, "ollama_status", lambda: (True, True))
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_ready"] is True


def test_health_degraded_without_ollama(client):
    r = client.get("/health")
    # still a 200: logins and postings work without Ollama, only a dead database is a 503
    assert r.status_code == 200
    body = r.json()
    assert body["database"] == "ok"
    assert body["chroma"] == "ok"
    assert body["ollama"] == "disconnected"
    assert body["status"] == "degraded"


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


def test_ask_handles_no_matching_documents(client, headers, fake_answer, monkeypatch):
    client.post("/jobs", headers=headers, json=POSTING)
    # a cutoff nothing can pass, same as a question with no related chunk
    monkeypatch.setattr(rag_pipeline.settings, "max_distance", 0.0)
    r = client.post("/ask", headers=headers, json={"question": "How do I bake sourdough bread?"})
    assert r.json()["answer"] == NO_MATCH_ANSWER
    assert r.json()["confidence"] == "low"
    assert r.json()["sources"] == [] and r.json()["model"] is None
    # the guardrail: nothing close enough means the model is never asked
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


def test_chunks_stay_under_the_limit_and_keep_headings():
    for sample in rag_pipeline.load_samples():
        chunks = rag_pipeline.chunk_posting(sample["text"])
        assert all(len(c) <= rag_pipeline.MAX_CHUNK_CHARS for c in chunks)
        # a heading is never the last line of a chunk, cut off from its section
        assert not any(c.splitlines()[-1].startswith("#") for c in chunks)


def test_named_company_limits_the_search():
    class Job:
        def __init__(self, id, company):
            self.id, self.company = id, company

    jobs = [Job(1, "Harbor & Pine Insurance"), Job(2, "Cobalt Freight"), Job(3, "AI Labs")]
    assert rag_pipeline.named_postings("What does Harbor & Pine want for SQL?", jobs) == [1]
    assert rag_pipeline.named_postings("what does cobalt want?", jobs) == [2]
    assert rag_pipeline.named_postings("Which jobs ask for Docker?", jobs) is None
    # "ai" is too short to count as naming AI Labs
    assert rag_pipeline.named_postings("Which AI jobs are remote?", jobs) is None


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("Python is required [1].", "high"),
        ("Python is required.", "low"),
        ("Python is required [7].", "low"),
        ("The saved postings don't have enough information to answer that.", "low"),
    ],
)
def test_confidence(answer, expected):
    chunks = [{"distance": 0.5}, {"distance": 0.9}]
    assert rag_pipeline.compute_confidence(chunks, answer) == expected
