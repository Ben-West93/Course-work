"""
Auth tests: registering, logging in, the token, and protected routes
rejecting requests without a valid one. No Postgres, Ollama or Docker needed:

    cd backend && pytest tests/test_auth.py -v

To run them against Postgres, see TEST_DATABASE_URL in conftest.py.
"""

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

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


@pytest.mark.parametrize(
    "auth_header",
    [
        None,
        "Bearer not-a-jwt",
        # signed with a different secret
        "Bearer "
        + jwt.encode({"sub": "1", "exp": 4102444800}, "a-different-secret-of-at-least-32-bytes", algorithm="HS256"),
        # right secret, already expired
        "Bearer " + jwt.encode({"sub": "1", "exp": 1}, os.environ["JWT_SECRET_KEY"], algorithm="HS256"),
    ],
)
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


def test_a_deleted_accounts_token_never_opens_the_next_account(client):
    # SQLite gave the next account the deleted one's id, and the old token,
    # still signed and unexpired, signed in as that new account
    first = signup(client)
    first_id = client.get("/me", headers=first).json()["id"]
    assert client.delete("/me", headers=first).status_code == 204
    second = signup(client)
    assert client.get("/me", headers=second).json()["id"] != first_id
    assert client.get("/me", headers=first).status_code == 401


def test_token_issued_before_the_account_existed_is_401(client):
    user_id = client.get("/me", headers=signup(client)).json()["id"]
    exp = int(time.time()) + 300

    def bearer(**claims) -> dict:
        return {"Authorization": "Bearer " + jwt.encode(claims, auth.SECRET_KEY, algorithm=auth.ALGORITHM)}

    # an old token for this id, as one looks after `docker-compose down -v` restarted the ids
    assert client.get("/me", headers=bearer(sub=str(user_id), iat=1, exp=exp)).status_code == 401
    # without iat there's nothing to check it against
    assert client.get("/me", headers=bearer(sub=str(user_id), exp=exp)).status_code == 401
    # the same claims issued now are accepted
    assert client.get("/me", headers=bearer(sub=str(user_id), iat=int(time.time()), exp=exp)).status_code == 200


def login(client, username: str, password: str):
    """POST /auth/token with the form fields, the way Swagger and the UI send them."""
    return client.post("/auth/token", data={"username": username, "password": password})


def test_five_wrong_passwords_block_the_username_for_15_minutes(client, monkeypatch):
    name = f"guess_{uuid.uuid4().hex[:8]}"
    client.post("/auth/register", json={"username": name, "password": "the-right-one"})
    for _ in range(5):
        assert login(client, name, "a-wrong-one").status_code == 401

    blocked = login(client, name, "a-wrong-one")
    assert blocked.status_code == 429
    assert blocked.json()["detail"] == "Too many failed sign-ins for this username. Try again in 15 minutes."
    assert 0 < int(blocked.headers["Retry-After"]) <= 15 * 60
    # the right password is refused too, or the block would only slow guessing down. Any case
    assert login(client, name.upper(), "the-right-one").status_code == 429
    # nobody else is affected
    other = f"bystander_{uuid.uuid4().hex[:8]}"
    client.post("/auth/register", json={"username": other, "password": "correct-horse"})
    assert login(client, other, "correct-horse").status_code == 200

    # the attempts while blocked didn't count, so it opens 15 minutes after the first failure
    later = auth.clock() + 15 * 60 + 1
    monkeypatch.setattr(auth, "clock", lambda: later)
    assert login(client, name, "the-right-one").status_code == 200


def test_unknown_usernames_are_blocked_the_same_way(client):
    # otherwise a 429 for some names and not others would show which ones exist
    name = f"nobody_{uuid.uuid4().hex[:8]}"
    assert [login(client, name, "whatever1").status_code for _ in range(6)] == [401] * 5 + [429]


def test_signing_in_clears_the_count(client):
    name = f"forgetful_{uuid.uuid4().hex[:8]}"
    client.post("/auth/register", json={"username": name, "password": "the-right-one"})
    for _ in range(4):
        login(client, name, "a-wrong-one")
    assert login(client, name, "the-right-one").status_code == 200
    assert [login(client, name, "a-wrong-one").status_code for _ in range(4)] == [401] * 4


def test_tracked_usernames_are_pruned(monkeypatch):
    # random usernames can't grow the table forever: expired ones go once it's full
    monkeypatch.setattr(auth, "failed_logins", {f"old_{i}": [0.0] for i in range(3)})
    monkeypatch.setattr(auth, "FAILED_LOGIN_KEYS", 3)
    monkeypatch.setattr(auth, "clock", lambda: auth.FAILED_LOGIN_WINDOW + 1.0)
    auth.start_login_attempt("new_one")
    assert list(auth.failed_logins) == ["new_one"]


def test_each_attempt_counts_before_its_password_is_checked(monkeypatch):
    # counted after the check, 19 of 40 guesses sent at once got through: all
    # of them passed the check before the first wrong one was counted
    monkeypatch.setattr(auth, "failed_logins", {})
    starts = [auth.start_login_attempt("burst") for _ in range(8)]
    assert starts[:5] == [0] * 5 and all(s > 0 for s in starts[5:])
    # the ones turned away weren't counted, so it still opens 15 minutes after the first
    assert len(auth.failed_logins["burst"]) == 5


def test_guesses_sent_at_once_get_five_tries_between_them(client):
    name = f"burst_{uuid.uuid4().hex[:8]}"
    client.post("/auth/register", json={"username": name, "password": "the-right-one"})
    with ThreadPoolExecutor(20) as pool:
        codes = list(pool.map(lambda i: login(client, name, f"guess-{i:02d}").status_code, range(20)))
    assert sorted(codes) == [401] * 5 + [429] * 15
