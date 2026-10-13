"""
Auth tests: registering, logging in, the token, and protected routes
rejecting requests without a valid one. No Postgres, Ollama or Docker needed:

    cd backend && pytest tests/test_auth.py -v

To run them against Postgres, see TEST_DATABASE_URL in conftest.py.
"""

import os

import jwt
import pytest

import auth
import rag_pipeline
from database import SessionLocal
from models import JobPosting
from tests.conftest import POSTING, signup


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


def test_passwords_are_stored_hashed():
    hashed = auth.hash_password("correct-horse")
    assert "correct-horse" not in hashed and hashed.startswith("$argon2")
    assert auth.verify_password("correct-horse", hashed)
    assert not auth.verify_password("Correct-horse", hashed)


@pytest.mark.parametrize("auth_header", [
    None,
    "Bearer not-a-jwt",
    # signed with a different secret
    "Bearer " + jwt.encode({"sub": "1", "exp": 4102444800}, "a-different-secret-of-at-least-32-bytes", algorithm="HS256"),
    # right secret, already expired
    "Bearer " + jwt.encode({"sub": "1", "exp": 1}, os.environ["JWT_SECRET_KEY"], algorithm="HS256"),
])
def test_protected_routes_reject_bad_tokens(client, auth_header):
    headers = {"Authorization": auth_header} if auth_header else {}
    for method, path in [("get", "/jobs"), ("post", "/ask"), ("get", "/me")]:
        r = client.request(method, path, headers=headers, json={"question": "anything?"})
        assert r.status_code == 401, (method, path)


def test_tokens_use_the_configured_algorithm(client, monkeypatch):
    # what JWT_ALGORITHM=HS512 with a 64-byte JWT_SECRET_KEY sets up at startup
    monkeypatch.setattr(auth, "ALGORITHM", "HS512")
    monkeypatch.setattr(auth, "SECRET_KEY", "k" * 64)
    headers = signup(client)
    token = headers["Authorization"].removeprefix("Bearer ")
    assert jwt.get_unverified_header(token)["alg"] == "HS512"
    assert client.get("/me", headers=headers).status_code == 200

    # the same secret and claims under another algorithm is refused: the API
    # never takes the algorithm from the token's own header
    claims = jwt.decode(token, "k" * 64, algorithms=["HS512"])
    downgraded = jwt.encode(claims, "k" * 64, algorithm="HS256")
    assert client.get("/me", headers={"Authorization": f"Bearer {downgraded}"}).status_code == 401


def test_deleting_account_cascades(client):
    headers = signup(client)
    job = client.post("/jobs", headers=headers, json=POSTING).json()
    assert client.delete("/me", headers=headers).status_code == 204
    with SessionLocal() as db:
        assert db.get(JobPosting, job["id"]) is None
    assert rag_pipeline.collection.get(where={"job_id": job["id"]})["ids"] == []
    # the token is still signed and unexpired, but the user is gone
    assert client.get("/me", headers=headers).status_code == 401
