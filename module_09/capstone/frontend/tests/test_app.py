"""
UI tests: runs app.py with Streamlit's AppTest against a fake API
(conftest.py), no backend needed.

    cd frontend && pip install -r requirements-dev.txt && pytest -q

Each test sets the API's replies, runs the script, and checks what would be
on screen or in session state.
"""

import ast
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

import api_client
from api_client import BACKEND_URL
from tests.conftest import HEALTH, JOB, Reply

APP = str(Path(__file__).resolve().parent.parent / "app.py")
HOSTILE_TITLE = "![x](http://evil.example/t.png) $90k-$110k *Lead*"


def app(signed_in: bool = True) -> AppTest:
    """app.py run once against the fake API, signed in unless told otherwise."""
    at = AppTest.from_file(APP, default_timeout=10)
    if signed_in:
        at.session_state["token"] = "token"
        at.session_state["username"] = "tester"
    return at.run()


def ask(at: AppTest, question: str) -> dict:
    """Send a question from the chat box. Returns the reply it added to the chat."""
    at.chat_input[0].set_value(question).run()
    return at.session_state["chat"][-1]


def all_text(at: AppTest) -> str:
    """Every message, caption and title on the page and in the sidebar, as one string."""
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


@pytest.mark.parametrize("failure", [requests.exceptions.ConnectionError(), requests.exceptions.ReadTimeout()])
@pytest.mark.parametrize("form", [0, 1], ids=["sign-in", "create-account"])
def test_api_going_away_during_sign_in_is_an_error_not_a_crash(fake_api, form, failure):
    # /health answered, then the API went away before the form was sent.
    # This was a ConnectionError traceback on the sign-in page
    fake_api[("POST", "/auth/token")] = failure
    fake_api[("POST", "/auth/register")] = failure
    at = app(signed_in=False)
    at.text_input[2 * form].set_value("someone")
    at.text_input[2 * form + 1].set_value("password123")
    at.button[form].click().run()
    assert not at.exception
    assert any(BACKEND_URL in e.value for e in at.error)
    assert at.session_state["token"] is None


def test_account_created_but_sign_in_failed_says_so(fake_api):
    fake_api[("POST", "/auth/register")] = Reply(
        201, {"id": 1, "username": "someone", "created_at": "2026-10-13T00:00:00Z"}
    )
    fake_api[("POST", "/auth/token")] = Reply(503, {"detail": "Database unavailable"})
    at = app(signed_in=False)
    at.text_input[2].set_value("someone")
    at.text_input[3].set_value("password123")
    at.button[1].click().run()
    assert not at.exception
    assert any("Account created, but signing in failed: Database unavailable" in e.value for e in at.error)


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
        return Reply(
            200,
            {
                "answer": "The internship pays $32 per hour [1].",
                "confidence": "high",
                "truncated": False,
                "model": "llama3.2:3b",
                "sources": [
                    {
                        "n": 1,
                        "job_id": 1,
                        "title": "Dev",
                        "company": "Same Co",
                        "chunk": 4,
                        "excerpt": "## Pay\n\n$32 per hour, 40 hours a week.",
                        "distance": 0.66,
                    }
                ],
            },
        )

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


@pytest.mark.parametrize(
    "reply, kind, text",
    [
        (requests.exceptions.ConnectionError(), "error", "Lost the connection"),
        (requests.exceptions.ReadTimeout(), "error", "The model may still be loading"),
        (Reply(200, None, "<html>proxy</html>"), "error", "wasn't JSON"),
        (
            Reply(422, {"detail": [{"loc": ["body", "question"], "msg": "String should have at least 3 characters"}]}),
            "warning",
            "question: String should have at least 3 characters",
        ),
        (Reply(503, {"detail": "Ollama returned 404 for model llama3.2:3b"}), "error", "ollama-pull"),
        (Reply(503, {"detail": "Ollama unavailable"}), "error", "docker-compose ps ollama"),
        (Reply(503, {"detail": "Database unavailable"}), "error", "docker-compose ps postgres"),
    ],
)
def test_every_failure_becomes_a_reply_in_the_chat(fake_api, reply, kind, text):
    fake_api[("POST", "/ask")] = reply
    at = app()
    answer = ask(at, "What is required?")
    assert not at.exception
    assert answer["kind"] == kind and text in answer["content"]
    # the question stays answered, with the error as the reply
    assert [m["role"] for m in at.session_state["chat"]] == ["user", "assistant"]


def test_links_in_answers_are_plain_text(fake_api):
    fake_api[("POST", "/ask")] = Reply(
        200,
        {
            "answer": "Visit http://shady.example or www.shady.example, or [apply here](https://shady.example/a).",
            "confidence": "low",
            "truncated": False,
            "model": "m",
            "sources": [],
        },
    )
    at = app()
    ask(at, "What does Shady Corp pay?")
    rendered = " ".join(m.value for m in at.markdown)
    assert "http:\u2060//shady.example" in rendered
    assert "www\u2060.shady.example" in rendered
    assert "[apply here]\\(https:\u2060//" in rendered


def test_hostile_answer_and_truncation(fake_api):
    fake_api[("POST", "/ask")] = Reply(
        200,
        {
            "answer": "![x](http://evil.example/a.png) pays $90k [1] and",
            "confidence": "low",
            "truncated": True,
            "model": "m",
            "sources": [
                {
                    "n": 1,
                    "job_id": 1,
                    "title": "*Dev*",
                    "company": "Same Co",
                    "chunk": 0,
                    "excerpt": "$90k",
                    "distance": 0.5,
                }
            ],
        },
    )
    at = app()
    ask(at, "What does it pay?")
    rendered = " ".join(m.value for m in at.markdown)
    assert "!\\[x]" in rendered and "\\$90k" in rendered
    assert "\\*Dev\\*" in rendered
    assert any("cut off" in c.value for c in at.caption)


# ── Jobs page ──────────────────────────────────────────────────────────────


def jobs_page(at: AppTest) -> AppTest:
    """Switch to the Jobs page in the sidebar."""
    return at.sidebar.radio[0].set_value("Jobs").run()


def test_hostile_titles_are_shown_as_text(fake_api):
    fake_api[("GET", "/jobs")] = Reply(200, [dict(JOB, title=HOSTILE_TITLE, company="C++ Labs", indexed=False)])
    at = jobs_page(app())
    row = next(m.value for m in at.markdown if "evil" in m.value)
    assert "\\!\\[x\\]" in row and "\\$90k" in row and "\\*Lead\\*" in row
    # "http://" can't turn into a link
    assert "http\\:⁠//" in row
    assert any("not indexed" in c.value for c in at.caption)


def test_hostile_locations_are_shown_as_text(fake_api):
    # the location is under the title in st.caption, which renders Markdown as well
    fake_api[("GET", "/jobs")] = Reply(200, [dict(JOB, location=HOSTILE_TITLE)])
    at = jobs_page(app())
    caption = next(c.value for c in at.caption if "evil" in c.value)
    assert "\\!\\[x\\]" in caption and "\\$90k" in caption and "\\*Lead\\*" in caption
    assert "http\\:⁠//" in caption


def test_failed_save_keeps_the_form_and_success_clears_it(fake_api):
    state = {"status": 422}

    def save(**_kw):
        if state["status"] == 422:
            return Reply(
                422, {"detail": [{"loc": ["body", "text"], "msg": "String should have at least 200 characters"}]}
            )
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


def test_a_saved_upload_leaves_an_empty_upload_box(fake_api):
    # the uploader used to keep the saved file, so the text box stayed locked
    # and the next posting sent the same file again (a 409)
    sent = []
    fake_api[("POST", "/jobs")] = lambda json=None, **_kw: (
        sent.append(json) or Reply(201, {"id": 9, "title": "Dev", "company": "Kestrel Robotics", "indexed": True})
    )
    at = jobs_page(app())
    at.file_uploader[0].set_value(("posting.txt", b"Kestrel Robotics is hiring. " * 20, "text/plain")).run()
    assert at.text_area(key="job_text").proto.disabled
    at.text_input(key="job_title").set_value("Dev")
    at.text_input(key="job_company").set_value("Kestrel Robotics")
    next(b for b in at.button if b.label == "Save posting").click().run()

    assert sent[0]["source"] == "upload" and sent[0]["text"].startswith("Kestrel Robotics is hiring.")
    assert at.file_uploader[0].value is None
    assert not at.text_area(key="job_text").proto.disabled


def test_jobs_page_counts_postings_companies_and_indexed(fake_api):
    fake_api[("GET", "/jobs")] = Reply(
        200, [JOB, dict(JOB, id=2, company="same co"), dict(JOB, id=3, company="Other Co", indexed=False)]
    )
    at = jobs_page(app())
    assert [(m.label, m.value) for m in at.metric] == [
        ("Postings", "3"),
        ("Companies", "2"),
        ("Indexed for search", "2 of 3"),
    ]
    # nothing saved: the "Nothing saved yet" note instead of three zeros
    fake_api[("GET", "/jobs")] = Reply(200, [])
    assert not jobs_page(app()).metric


def test_postings_past_the_first_hundred_are_listed(fake_api):
    def route(params, **_kw):
        return Reply(200, [dict(JOB, id=i) for i in range(params["offset"], min(params["offset"] + 100, 130))])

    fake_api[("GET", "/jobs")] = route
    at = jobs_page(app())
    # one Yes, delete button per posting, so all 130 are on the page
    assert sum(b.label == "Yes, delete" for b in at.button) == 130
    assert not any("newest" in c.value for c in at.caption)


def test_a_posting_repeated_across_pages_is_shown_once(fake_api):
    # two Delete buttons with one key made the whole Jobs page fail
    def route(params, **_kw):
        start = 0 if params["offset"] == 0 else 99
        return Reply(200, [dict(JOB, id=i) for i in range(start, 100 if start == 0 else 130)])

    fake_api[("GET", "/jobs")] = route
    at = jobs_page(app())
    assert not at.exception
    assert sum(b.label == "Yes, delete" for b in at.button) == 130


def test_posting_cap_is_shown(fake_api, monkeypatch):
    monkeypatch.setattr(api_client, "MAX_POSTINGS", 150)
    fake_api[("GET", "/jobs")] = lambda params, **_kw: Reply(
        200, [dict(JOB, id=params["offset"] + i) for i in range(100)]
    )
    at = jobs_page(app())
    assert any("Showing your 150 newest postings" in c.value for c in at.caption)


def test_deleting_takes_a_second_click(fake_api):
    deleted = []
    fake_api[("DELETE", "/jobs/1")] = lambda **_kw: deleted.append(1) or Reply(204)
    at = jobs_page(app())
    # Delete only opens the question, nothing is sent until Yes, delete
    assert "Delete this posting? It can't be undone." in all_text(at)
    assert not any(b.label == "Delete" for b in at.button) and deleted == []
    next(b for b in at.button if b.label == "Yes, delete").click().run()
    assert deleted == [1]
    assert any("Deleted Dev" in s.value for s in at.success)


def test_postings_failing_to_load_is_an_error_on_both_pages(fake_api):
    fake_api[("GET", "/jobs")] = Reply(503, {"detail": "Database unavailable"})
    at = app()
    assert any("Couldn't load your postings for the filter: Database unavailable" in e.value for e in at.error)
    # not told to save a posting they may well have
    assert "Save a posting" not in all_text(at)
    at = jobs_page(at)
    assert any("Couldn't load postings: Database unavailable" in e.value for e in at.error)
    assert not at.exception


@pytest.mark.parametrize(
    "route, label",
    [
        (("DELETE", "/jobs/1"), "Yes, delete"),
        (("POST", "/jobs/samples"), "Load sample postings"),
        (("POST", "/jobs/reindex"), "Rebuild search index"),
    ],
)
def test_jobs_page_actions_with_the_api_gone_show_an_error(fake_api, route, label):
    fake_api[route] = requests.exceptions.ConnectionError()
    at = jobs_page(app())
    next(b for b in at.button if b.label == label).click().run()
    assert not at.exception
    assert any("Lost the connection" in e.value for e in at.error)
    assert at.session_state["token"] == "token"


def test_saving_when_the_api_times_out_keeps_the_form(fake_api):
    fake_api[("POST", "/jobs")] = requests.exceptions.ReadTimeout()
    at = jobs_page(app())
    at.text_input(key="job_title").set_value("Dev")
    at.text_area(key="job_text").set_value("x" * 300)
    next(b for b in at.button if b.label == "Save posting").click().run()
    assert not at.exception
    assert any("after 60s" in e.value for e in at.error)
    assert at.text_area(key="job_text").value == "x" * 300


def test_expired_session_during_an_action_goes_back_to_the_login_page(fake_api):
    fake_api[("DELETE", "/jobs/1")] = Reply(401, {"detail": "Could not validate credentials"})
    at = jobs_page(app())
    next(b for b in at.button if b.label == "Yes, delete").click().run()
    assert at.session_state["token"] is None
    assert "session expired" in all_text(at).lower()


def test_app_makes_no_http_calls_of_its_own():
    # every call goes through api_client.py, as the course starter asks
    tree = ast.parse(Path(APP).read_text())
    imported = {
        alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    }
    imported |= {
        node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "api_client" in imported
    assert not imported & {"requests", "httpx", "urllib", "urllib3", "http", "aiohttp"}
