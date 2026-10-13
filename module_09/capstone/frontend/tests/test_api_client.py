"""
api_client.py on its own, against the fake API in conftest.py: what each
call sends, what it returns, and that every failure comes out as ApiError or
SessionExpired, never a requests exception.

    cd frontend && pytest tests/test_api_client.py -v
"""

import pytest
import requests

import api_client
from api_client import ASK_TIMEOUT, BACKEND_URL, ApiError, ApiTimeout, SessionExpired
from tests.conftest import HEALTH, Reply


def recorder(reply: Reply, seen: dict):
    """A route that remembers the keyword arguments it was called with."""
    def route(**kwargs):
        seen.update(kwargs)
        return reply
    return route


# ── request(): headers, status codes, failures ────────────────────────────

def test_token_goes_in_the_bearer_header(fake_api):
    seen = {}
    fake_api[("GET", "/jobs")] = recorder(Reply(200, []), seen)
    assert api_client.list_jobs("abc") == []
    assert seen["headers"] == {"Authorization": "Bearer abc"}
    assert seen["params"] == {"limit": 100} and seen["timeout"] == 30


def test_no_token_means_no_authorization_header(fake_api):
    seen = {}
    fake_api[("POST", "/auth/token")] = recorder(Reply(200, {"access_token": "t", "token_type": "bearer"}), seen)
    assert api_client.login("ben", "pw123456") == "t"
    assert seen["headers"] == {}
    # OAuth2's password flow is a form, not JSON
    assert seen["data"] == {"username": "ben", "password": "pw123456"} and "json" not in seen


def test_401_with_a_token_is_an_expired_session(fake_api):
    fake_api[("GET", "/jobs")] = Reply(401, {"detail": "Could not validate credentials"})
    with pytest.raises(SessionExpired):
        api_client.list_jobs("old-token")


def test_401_without_a_token_is_the_apis_message(fake_api):
    fake_api[("POST", "/auth/token")] = Reply(401, {"detail": "Incorrect username or password"})
    with pytest.raises(ApiError) as e:
        api_client.login("ben", "wrong-password")
    assert (str(e.value), e.value.status_code) == ("Incorrect username or password", 401)


@pytest.mark.parametrize("reply, message", [
    (Reply(409, {"detail": "Username already taken"}), "Username already taken"),
    (Reply(422, {"detail": [{"loc": ["body", "password"], "msg": "String should have at least 8 characters"},
                            {"loc": ["body"], "msg": "Value error, bad"}]}),
     "password: String should have at least 8 characters; bad"),
    (Reply(502, None, "<html>Bad Gateway</html>"), "<html>Bad Gateway</html>"),
    (Reply(500, None, ""), "HTTP 500"),
    (Reply(500, ["not", "a", "dict"]), "HTTP 500"),
])
def test_error_replies_become_a_readable_message(fake_api, reply, message):
    fake_api[("POST", "/auth/register")] = reply
    with pytest.raises(ApiError) as e:
        api_client.register("ben", "pw")
    assert str(e.value) == message
    assert e.value.status_code == reply.status_code


def test_204_returns_none(fake_api):
    fake_api[("DELETE", "/jobs/5")] = Reply(204, None, "")
    assert api_client.delete_job("t", 5) is None


def test_non_json_success_is_an_error_with_no_status(fake_api):
    fake_api[("GET", "/jobs")] = Reply(200, None, "<html>captive portal</html>")
    with pytest.raises(ApiError) as e:
        api_client.list_jobs("t")
    assert "wasn't JSON (HTTP 200)" in str(e.value) and e.value.status_code is None


def test_timeout_is_an_api_timeout_naming_the_wait(fake_api):
    fake_api[("POST", "/jobs/samples")] = requests.exceptions.ReadTimeout()
    with pytest.raises(ApiTimeout) as e:
        api_client.add_samples("t")
    assert str(e.value) == f"No answer from `{BACKEND_URL}` after 120s."
    assert isinstance(e.value, ApiError) and e.value.status_code is None


CALLS = [
    (("POST", "/auth/token"), lambda: api_client.login("u", "p")),
    (("POST", "/auth/register"), lambda: api_client.register("u", "p")),
    (("GET", "/jobs"), lambda: api_client.list_jobs("t")),
    (("POST", "/jobs"), lambda: api_client.create_job("t", {"title": "x"})),
    (("DELETE", "/jobs/1"), lambda: api_client.delete_job("t", 1)),
    (("POST", "/jobs/samples"), lambda: api_client.add_samples("t")),
    (("POST", "/jobs/reindex"), lambda: api_client.reindex("t")),
    (("POST", "/ask"), lambda: api_client.ask("q?", "t")),
]


@pytest.mark.parametrize("route, call", CALLS, ids=[f"{m} {p}" for (m, p), _ in CALLS])
@pytest.mark.parametrize("failure", [
    requests.exceptions.ConnectionError(), requests.exceptions.ReadTimeout(), requests.exceptions.ConnectTimeout(),
    requests.exceptions.ChunkedEncodingError(), requests.exceptions.InvalidURL(), requests.exceptions.TooManyRedirects(),
], ids=lambda e: type(e).__name__)
def test_every_call_turns_network_failures_into_api_error(fake_api, route, call, failure):
    fake_api[route] = failure
    with pytest.raises(ApiError) as e:
        call()
    assert BACKEND_URL in str(e.value) and e.value.status_code is None


# ── health_check() ─────────────────────────────────────────────────────────

def test_health_check_returns_the_report(fake_api):
    assert api_client.health_check() == HEALTH


def test_health_check_keeps_the_report_when_the_database_is_down(fake_api):
    # /health is a 503 then, but its body is what the sidebar shows
    down = dict(HEALTH, status="down", database="error")
    fake_api[("GET", "/health")] = Reply(503, down)
    assert api_client.health_check() == down


@pytest.mark.parametrize("reply", [
    requests.exceptions.ConnectionError(), requests.exceptions.ReadTimeout(),
    Reply(502, None, "Bad Gateway"), Reply(200, ["a", "list"]),
])
def test_health_check_is_none_when_nothing_usable_answered(fake_api, reply):
    fake_api[("GET", "/health")] = reply
    assert api_client.health_check() is None


# ── ask() ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("job_ids, body", [
    (None, {"question": "What pays best?"}),
    ([], {"question": "What pays best?"}),
    ([3, 7], {"question": "What pays best?", "job_ids": [3, 7]}),
])
def test_ask_sends_job_ids_only_when_picked(fake_api, job_ids, body):
    seen = {}
    answer = {"answer": "x", "confidence": "low", "sources": [], "model": None}
    fake_api[("POST", "/ask")] = recorder(Reply(200, answer), seen)
    assert api_client.ask("What pays best?", "t", job_ids) == answer
    assert seen["json"] == body
    # longer than the backend's own 120s wait for Ollama
    assert seen["timeout"] == ASK_TIMEOUT > 120
