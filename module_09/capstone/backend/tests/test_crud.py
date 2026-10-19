"""
CRUD tests for job postings: create, list, get, update, delete, the bundled
samples, and one user never reaching another's postings.

    cd backend && pytest tests/test_crud.py -v

To run them against Postgres, see TEST_DATABASE_URL in conftest.py.
"""

import rag_pipeline
from main import NO_POSTINGS_ANSWER
from tests.conftest import POSTING, signup


def test_create_posting_indexes_its_chunks(client, headers):
    r = client.post("/jobs", headers=headers, json=POSTING)
    assert r.status_code == 201
    job = r.json()
    assert job["indexed"] is True
    stored = rag_pipeline.collection.get(where={"job_id": job["id"]})
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


def test_list_and_get_postings(client):
    headers = signup(client)
    assert client.get("/jobs", headers=headers).json() == []
    ids = [
        client.post(
            "/jobs", headers=headers, json=dict(POSTING, title=f"Role {i}", text=f"{POSTING['text']}\nref {i}")
        ).json()["id"]
        for i in range(3)
    ]

    listed = client.get("/jobs", headers=headers).json()
    # newest first, and the list leaves the full text out
    assert [j["id"] for j in listed] == ids[::-1]
    assert [j["title"] for j in listed] == ["Role 2", "Role 1", "Role 0"]
    assert all("text" not in j for j in listed)

    page = client.get("/jobs", headers=headers, params={"limit": 2, "offset": 1}).json()
    assert [j["id"] for j in page] == [ids[1], ids[0]]

    one = client.get(f"/jobs/{ids[0]}", headers=headers).json()
    assert one["title"] == "Role 0" and one["text"].endswith("ref 0")
    assert one["indexed"] is True and one["indexed_at"]


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
    docs = rag_pipeline.collection.get(where={"job_id": job["id"]})["documents"]
    assert docs and all(" at Renamed Freight (Remote)" in d.split("\n", 1)[0] for d in docs)

    assert client.patch(f"/jobs/{job['id']}", headers=headers, json={"title": None}).status_code == 422

    assert client.delete(f"/jobs/{job['id']}", headers=headers).status_code == 204
    assert rag_pipeline.collection.get(where={"job_id": job["id"]})["ids"] == []
    assert client.get(f"/jobs/{job['id']}", headers=headers).status_code == 404


def test_samples_load_once(client, headers):
    first = client.post("/jobs/samples", headers=headers).json()
    assert first["added"] == len(rag_pipeline.load_samples()) > 0
    assert first["skipped"] == 0
    second = client.post("/jobs/samples", headers=headers).json()
    assert second == {"added": 0, "skipped": first["added"]}
