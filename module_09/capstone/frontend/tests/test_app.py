"""
UI tests: runs app.py with Streamlit's AppTest against a fake API, no
backend needed.

    cd frontend && pip install -r requirements-dev.txt && pytest -q

`fake_api` swaps requests.request/get/post for a router keyed on
(method, path), which is all app.py uses. Each test sets the API's replies,
runs the script, and checks what would be on screen or in session state.
"""

import json
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
HEALTH = {"status": "ok", "database": "ok", "chroma": "ok", "ollama": "connected", "model_ready": True,
          "model": "llama3.2:3b"}
JOB = {"id": 1, "title": "Dev", "company": "Same Co", "location": None, "source": "pasted", "indexed": True}
HOSTILE_TITLE = "![x](http://evil.example/t.png) $90k-$110k *Lead*"


class Reply:
    def __init__(self, status_code=200, body=None, text=None):
        self.status_code, self._body = status_code, body
        self.text = text if text is not None else json.dumps(body)

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


@pytest.fixture
def fake_api(monkeypatch):
    """
    Returns the routes dict. A value is a Reply, an exception to raise, or a
    callable that gets the request's keyword arguments (json=, data=, ...).
    """
    routes = {("GET", "/health"): Reply(200, HEALTH), ("GET", "/jobs"): Reply(200, [JOB])}

    def handler(method, url, **kwargs):
        path = url.split("://", 1)[1].split("/", 1)[1].split("?")[0]
        reply = routes.get((method.upper(), f"/{path}"), Reply(404, {"detail": "no such route"}))
        if isinstance(reply, Exception):
            raise reply
        return reply(**kwargs) if callable(reply) else reply

    monkeypatch.setattr(requests, "request", handler)
    monkeypatch.setattr(requests, "get", lambda url, **kw: handler("GET", url, **kw))
    monkeypatch.setattr(requests, "post", lambda url, **kw: handler("POST", url, **kw))
    return routes


def app(signed_in: bool = True) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=10)
    if signed_in:
        at.session_state["token"] = "token"
        at.session_state["username"] = "tester"
    return at.run()


def ask(at: AppTest, question: str) -> dict:
    at.chat_input[0].set_value(question).run()
    return at.session_state["chat"][-1]


def all_text(at: AppTest) -> str:
    parts = []
    for root in (at, at.sidebar):
        for kind in ("markdown", "error", "warning", "info", "success", "caption", "title"):
            parts += [str(e.value) for e in getattr(root, kind)]
    return "\n".join(parts)


# ── Connection and session ─────────────────────────────────────────────────

def test_backend_down_is_an_error_not_a_crash(fake_api):
    fake_api[("GET", "/health")] = requests.exceptions.ConnectionError()
    at = app(signed_in=False)
    assert not at.exception
    assert "Can't reach the API" in all_text(at)


def test_non_json_health_counts_as_unreachable(fake_api):
    fake_api[("GET", "/health")] = Reply(502, None, "Bad Gateway")
    at = app(signed_in=False)
    assert not at.exception and "Can't reach the API" in all_text(at)


def test_wrong_password_shows_the_apis_message(fake_api):
    fake_api[("POST", "/auth/token")] = Reply(401, {"detail": "Incorrect username or password"})
    at = app(signed_in=False)
    at.text_input[0].set_value("someone")
    at.text_input[1].set_value("wrong-password")
    at.button[0].click().run()
    assert "Incorrect username or password" in all_text(at)
    assert at.session_state["token"] is None


def test_expired_session_goes_back_to_the_login_page(fake_api):
    fake_api[("GET", "/jobs")] = Reply(401, {"detail": "Could not validate credentials"})
    at = app()
    assert at.session_state["token"] is None
    assert "session expired" in all_text(at).lower()
    # the login tabs, and no stale "Signed in as" in the sidebar
    assert len(at.tabs) == 2
    assert not any("Signed in" in m.value for m in at.sidebar.markdown)


def test_model_still_downloading_is_shown(fake_api):
    fake_api[("GET", "/health")] = Reply(200, dict(HEALTH, status="degraded", model_ready=False))
    at = app()
    assert any("still downloading" in w.value for w in at.sidebar.warning)


# ── Ask page: the full flow from question to rendered answer ──────────────

def test_question_to_rendered_answer(fake_api):
    sent = {}

    def answer(json=None, **_kw):
        sent.update(json)
        return Reply(200, {
            "answer": "The internship pays $32 per hour [1].", "confidence": "high", "truncated": False,
            "model": "llama3.2:3b",
            "sources": [{"n": 1, "job_id": 1, "title": "Dev", "company": "Same Co", "chunk": 4,
                         "excerpt": "## Pay\n\n$32 per hour, 40 hours a week.", "distance": 0.66}],
        })

    fake_api[("POST", "/ask")] = answer
    at = app()
    reply = ask(at, "  What does it pay?  ")

    # what the UI sent: the trimmed question, no job filter
    assert sent == {"question": "What does it pay?"}
    assert reply["kind"] == "answer" and reply["confidence"] == "high"
    rendered = " ".join(m.value for m in at.markdown)
    # "$" escaped so Streamlit doesn't read "$32 ... $" as LaTeX
    assert "pays \\$32 per hour [1]" in rendered
    assert ":green-badge[Confidence: high]" in rendered
    assert "**\\[1\\] Dev** · Same Co · distance 0.66" in rendered
    # the excerpt is shown as plain text, exactly as stored
    assert [t.value for t in at.text] == ["## Pay\n\n$32 per hour, 40 hours a week."]


def test_picked_postings_are_sent_as_job_ids(fake_api):
    seen = {}

    def answer(json=None, **_kw):
        seen.update(json)
        return Reply(200, {"answer": "x", "confidence": "low", "sources": [], "model": None})

    fake_api[("GET", "/jobs")] = Reply(200, [JOB, dict(JOB, id=2)])
    fake_api[("POST", "/ask")] = answer
    at = app()
    # two postings with the same title and company are both selectable
    assert len(at.multiselect[0].options) == 2
    at.multiselect[0].set_value([at.multiselect[0].options[1]]).run()
    ask(at, "What is required?")
    assert seen["job_ids"] == [2]


@pytest.mark.parametrize("reply, kind, text", [
    (requests.exceptions.ConnectionError(), "error", "Lost the connection"),
    (Reply(200, None, "<html>proxy</html>"), "error", "wasn't JSON"),
    (Reply(422, {"detail": [{"loc": ["body", "question"], "msg": "String should have at least 3 characters"}]}),
     "warning", "question: String should have at least 3 characters"),
    (Reply(503, {"detail": "Ollama returned 404 for model llama3.2:3b"}), "error", "ollama-pull"),
    (Reply(503, {"detail": "Ollama unavailable"}), "error", "docker-compose ps ollama"),
    (Reply(503, {"detail": "Database unavailable"}), "error", "docker-compose ps postgres"),
])
def test_every_failure_becomes_a_reply_in_the_chat(fake_api, reply, kind, text):
    fake_api[("POST", "/ask")] = reply
    at = app()
    answer = ask(at, "What is required?")
    assert not at.exception
    assert answer["kind"] == kind and text in answer["content"]
    # the question stays answered, with the error as the reply
    assert [m["role"] for m in at.session_state["chat"]] == ["user", "assistant"]


def test_links_in_answers_are_plain_text(fake_api):
    fake_api[("POST", "/ask")] = Reply(200, {
        "answer": "Visit http://shady.example or www.shady.example, or [apply here](https://shady.example/a).",
        "confidence": "low", "truncated": False, "model": "m", "sources": []})
    at = app()
    ask(at, "What does Shady Corp pay?")
    rendered = " ".join(m.value for m in at.markdown)
    assert "http:\u2060//shady.example" in rendered
    assert "www\u2060.shady.example" in rendered
    assert "[apply here]\\(https:\u2060//" in rendered


def test_hostile_answer_and_truncation(fake_api):
    fake_api[("POST", "/ask")] = Reply(200, {
        "answer": "![x](http://evil.example/a.png) pays $90k [1] and", "confidence": "low", "truncated": True,
        "model": "m", "sources": [{"n": 1, "job_id": 1, "title": "*Dev*", "company": "Same Co", "chunk": 0,
                                   "excerpt": "$90k", "distance": 0.5}]})
    at = app()
    ask(at, "What does it pay?")
    rendered = " ".join(m.value for m in at.markdown)
    assert "!\\[x]" in rendered and "\\$90k" in rendered
    assert "\\*Dev\\*" in rendered
    assert any("cut off" in c.value for c in at.caption)


# ── Jobs page ──────────────────────────────────────────────────────────────

def jobs_page(at: AppTest) -> AppTest:
    return at.sidebar.radio[0].set_value("Jobs").run()


def test_hostile_titles_are_shown_as_text(fake_api):
    fake_api[("GET", "/jobs")] = Reply(200, [dict(JOB, title=HOSTILE_TITLE, company="C++ Labs", indexed=False)])
    at = jobs_page(app())
    row = next(m.value for m in at.markdown if "evil" in m.value)
    assert "\\!\\[x\\]" in row and "\\$90k" in row and "\\*Lead\\*" in row
    # "http://" can't turn into a link
    assert "http\\:⁠//" in row
    assert any("not indexed" in c.value for c in at.caption)


def test_failed_save_keeps_the_form_and_success_clears_it(fake_api):
    state = {"status": 422}

    def save(**_kw):
        if state["status"] == 422:
            return Reply(422, {"detail": [{"loc": ["body", "text"], "msg": "String should have at least 200 characters"}]})
        return Reply(201, {"id": 9, "title": HOSTILE_TITLE, "company": "C++ Labs", "indexed": True})

    fake_api[("POST", "/jobs")] = save
    at = jobs_page(app())
    at.text_input(key="job_title").set_value(HOSTILE_TITLE)
    at.text_input(key="job_company").set_value("C++ Labs")
    at.text_area(key="job_text").set_value("too short")
    save_button = next(b for b in at.button if b.label == "Save posting")

    save_button.click().run()
    assert "text: String should have at least 200 characters" in all_text(at)
    assert at.text_area(key="job_text").value == "too short"
    assert at.text_input(key="job_title").value == HOSTILE_TITLE

    state["status"] = 201
    next(b for b in at.button if b.label == "Save posting").click().run()
    notice = " ".join(s.value for s in at.success)
    # the notice escapes the title too, it used to load the image
    assert "\\!\\[x\\]" in notice and "\\$90k" in notice
    assert at.text_input(key="job_title").value == "" and at.text_area(key="job_text").value == ""


def test_hundred_posting_limit_is_shown(fake_api):
    fake_api[("GET", "/jobs")] = Reply(200, [dict(JOB, id=i) for i in range(100)])
    at = jobs_page(app())
    assert any("100 newest" in c.value for c in at.caption)
