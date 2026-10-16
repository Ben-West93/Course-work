"""
Streamlit UI for JobFit: log in, save postings, ask questions about them.

    BACKEND_URL=http://localhost:8000 streamlit run app.py

In Docker Compose BACKEND_URL is http://backend:8000.

Pages and the API calls behind them:
    sign in / create account   POST /auth/token, POST /auth/register
    Jobs                       GET /jobs, POST /jobs, DELETE /jobs/{id},
                               POST /jobs/samples, POST /jobs/reindex
    Ask                        GET /jobs (for the filter), POST /ask
    sidebar, every page        GET /health

Every one of those calls goes through api_client.py. Its errors are shown
as messages on the page, never as a traceback.

Everything here runs on the Streamlit server, not in the browser, so the JWT
never reaches the browser. Anything that came from a user or the model is
escaped before it goes into st.markdown (plain() and safe_markdown()).
Tests: tests/test_app.py and tests/test_api_client.py.
"""

import re

import streamlit as st

import api_client
from api_client import BACKEND_URL, ApiError, ApiTimeout, SessionExpired

CONFIDENCE_COLOURS = {"high": "green", "medium": "orange", "low": "red"}
JOB_FORM_KEYS = ("job_title", "job_company", "job_location", "job_url", "job_text")

st.set_page_config(page_title="JobFit", page_icon="🧭", layout="centered")

# the JWT lives here, on the Streamlit server, and never reaches the browser
for key, default in [("token", None), ("username", None), ("chat", []), ("notice", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


def logout():
    """Forget the token, the username and the chat."""
    st.session_state.token = None
    st.session_state.username = None
    st.session_state.chat = []


def login(username: str, password: str) -> str | None:
    """None on success, otherwise the error to show."""
    try:
        token = api_client.login(username, password)
    # a wrong password, or the API going away between the health check and
    # the click. The second used to be a traceback on the sign-in page
    except ApiError as e:
        return str(e)
    st.session_state.token = token
    st.session_state.username = username.strip().lower()
    return None


# Markdown that would change how a posting's title or company renders
MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:])")
# invisible and zero width. GFM still turned "http://..." in a title into a
# link after the escapes, a word joiner inside "://" and "www." stops that
WORD_JOINER = "\u2060"


def plain(text: str) -> str:
    """User text shown inside st.markdown as literal characters."""
    # a title like "![x](http://tracker)" loaded an image, "$90k-$110k" turned into LaTeX
    escaped = MARKDOWN_SPECIAL.sub(r"\\\1", str(text))
    return escaped.replace("\\:", "\\:" + WORD_JOINER).replace("www\\.", "www" + WORD_JOINER + "\\.")


def safe_markdown(text: str) -> str:
    """Model output for st.markdown: formatting kept, nothing that loads or links."""
    # st.markdown treats $...$ as LaTeX ("$95,000 to $115,000"), and would
    # load any ![image](url) in an answer, which a pasted posting could abuse.
    # Links are made plain text too: a posting with planted instructions got
    # 3b to answer "Visit http://shady.example to apply", which rendered as a
    # clickable link. "[text](url)" loses its "(" and "://" gets a word joiner
    text = text.replace("$", "\\$").replace("![", "!\\[").replace("](", "]\\(")
    return text.replace("://", ":" + WORD_JOINER + "//").replace("www.", "www" + WORD_JOINER + ".")


# ── Sidebar ────────────────────────────────────────────────────────────────

def sidebar(health: dict | None):
    """The app's name, what /health reported and, once signed in, who and a sign-out button."""
    with st.sidebar:
        st.title("🧭 JobFit")
        if health is None:
            st.error("Backend: unreachable")
            st.caption(f"Nothing answered at `{BACKEND_URL}`.")
        else:
            label = {"ok": "success", "degraded": "warning"}.get(health.get("status"), "error")
            getattr(st, label)(f"Backend: {health.get('status', '?')}")
            st.caption(f"Database: {health.get('database')} · Chroma: {health.get('chroma')} · "
                       f"Ollama: {health.get('ollama')}")
            if health.get("ollama") == "connected" and not health.get("model_ready"):
                st.warning(f"{health.get('model')} is still downloading. Questions will fail until it's ready.")

        if st.session_state.token:
            st.divider()
            st.write(f"Signed in as **{plain(st.session_state.username)}**")
            st.button("Sign out", on_click=logout, width="stretch")


def login_page():
    """The sign-in and create-account tabs."""
    st.title("JobFit")
    st.caption("Save the job postings you're applying to and ask questions about them.")
    if st.session_state.pop("expired", False):
        st.warning("Your session expired. Sign in again.")
    sign_in, register = st.tabs(["Sign in", "Create account"])
    with sign_in, st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            error = login(username, password)
            if error:
                st.error(error)
            else:
                st.rerun()
    with register, st.form("register"):
        username = st.text_input("Username", help="3-50 characters: letters, digits, _ . -")
        password = st.text_input("Password", type="password", help="At least 8 characters")
        if st.form_submit_button("Create account", type="primary"):
            try:
                api_client.register(username, password)
            except ApiError as e:
                st.error(str(e))
            else:
                error = login(username, password)
                if error:
                    st.error(f"Account created, but signing in failed: {error}")
                else:
                    st.rerun()


# ── Jobs ───────────────────────────────────────────────────────────────────

def jobs_page():
    """The saved postings with their Delete buttons, the samples and reindex buttons, and the form to add one."""
    st.header("Saved postings")
    if st.session_state.notice:
        kind, message = st.session_state.notice
        getattr(st, kind)(message)
        st.session_state.notice = None

    try:
        jobs = api_client.list_jobs(st.session_state.token)
    except ApiError as e:
        st.error(f"Couldn't load postings: {e}")
        jobs = []

    if not jobs:
        st.info("Nothing saved yet. Add a posting below, or load the sample postings.")
    elif len(jobs) == 100:
        st.caption("Showing your 100 newest postings.")
    for job in jobs:
        with st.container(border=True):
            left, right = st.columns([5, 1])
            left.markdown(f"**{plain(job['title'])}** · {plain(job['company'])}")
            # st.caption renders Markdown too, and the location is the user's text
            location = plain(job["location"]) if job.get("location") else None
            left.caption(" · ".join(filter(None, [location, job["source"],
                                                  "indexed" if job["indexed"] else "not indexed"])))
            if right.button("Delete", key=f"del_{job['id']}"):
                try:
                    api_client.delete_job(st.session_state.token, job["id"])
                    st.session_state.notice = ("success", f"Deleted {plain(job['title'])}")
                except ApiError as e:
                    st.session_state.notice = ("error", str(e))
                st.rerun()

    cols = st.columns(2)
    if cols[0].button("Load sample postings", width="stretch"):
        try:
            with st.spinner("Saving and indexing the samples..."):
                s = api_client.add_samples(st.session_state.token)
            st.session_state.notice = ("success", f"Added {s['added']}, skipped {s['skipped']} already saved")
        except ApiError as e:
            st.session_state.notice = ("error", str(e))
        st.rerun()
    if cols[1].button("Rebuild search index", width="stretch", disabled=not jobs):
        try:
            with st.spinner("Re-indexing..."):
                failed = api_client.reindex(st.session_state.token)["failed"]
            st.session_state.notice = ("warning", f"Failed: {failed}") if failed else ("success", "All postings re-indexed")
        except ApiError as e:
            st.session_state.notice = ("error", str(e))
        st.rerun()

    st.subheader("Add a posting")
    # cleared only after a successful save. clear_on_submit wiped a pasted
    # posting on every 422 or 409 too. Widget values can't change once the
    # widget exists in this run, so the save sets a flag and this runs first
    if st.session_state.pop("clear_job_form", False):
        for key in JOB_FORM_KEYS:
            st.session_state[key] = ""
    upload = st.file_uploader("Upload a .txt or .md file, or paste the text below", type=["txt", "md"])
    with st.form("add_job"):
        title = st.text_input("Job title", key="job_title")
        company = st.text_input("Company", key="job_company")
        location = st.text_input("Location (optional)", key="job_location")
        url = st.text_input("Link (optional)", key="job_url")
        text = st.text_area("Posting text", height=200, help="200 to 20,000 characters", key="job_text",
                            disabled=upload is not None,
                            placeholder="Using the uploaded file" if upload else "Paste the full posting")
        if st.form_submit_button("Save posting", type="primary"):
            body = {"title": title, "company": company, "location": location, "url": url, "source": "pasted",
                    "text": text}
            if upload is not None:
                body["text"] = upload.getvalue().decode("utf-8", errors="replace")
                body["source"] = "upload"
            try:
                with st.spinner("Saving and indexing..."):
                    job = api_client.create_job(st.session_state.token, body)
            except ApiError as e:
                st.error(str(e))
            else:
                note = "" if job["indexed"] else " It isn't searchable yet, use Rebuild search index."
                # st.success renders Markdown too, the notice loaded a title's ![image](url)
                st.session_state.notice = ("success", f"Saved {plain(job['title'])} at {plain(job['company'])}.{note}")
                st.session_state.clear_job_form = True
                st.rerun()


# ── Ask ────────────────────────────────────────────────────────────────────

def failure_hint(detail: str) -> str:
    """What to check for each 503 the API sends, by its detail message."""
    if detail == "Database unavailable":
        return "Postgres isn't answering, check `docker-compose ps postgres`."
    if detail == "Ollama unavailable":
        return "Check `docker-compose ps ollama`."
    if detail.startswith("Ollama returned 404"):
        return "The model isn't pulled yet, check `docker-compose logs ollama-pull`."
    if "timed out" in detail:
        return "The model may still be loading, try again in a moment."
    return "The cause is in `docker-compose logs backend`."


def ask(question: str, job_ids: list[int]) -> dict:
    """The assistant's reply to a question. A failed request is a warning or error reply, never an exception."""
    try:
        data = api_client.ask(question, st.session_state.token, job_ids)
    # the question is already in the chat, so the failure has to be a reply
    # to it rather than a page error that leaves it unanswered
    except ApiTimeout as e:
        return {"kind": "error", "content": f"{e} The model may still be loading."}
    except ApiError as e:
        if e.status_code == 422:
            return {"kind": "warning", "content": str(e)}
        # no usable reply: lost connection, or not JSON
        if e.status_code is None:
            return {"kind": "error", "content": str(e)}
        return {"kind": "error", "content": f"**{e}** (HTTP {e.status_code}). {failure_hint(str(e))}"}
    return {"kind": "answer", "content": data["answer"], "confidence": data["confidence"],
            "sources": data["sources"], "model": data.get("model"), "truncated": data.get("truncated", False)}


def render(msg: dict):
    """Draw one chat message. An answer from the assistant also gets its confidence badge and sources."""
    with st.chat_message(msg["role"]):
        kind = msg.get("kind", "answer")
        if kind != "answer":
            getattr(st, kind)(msg["content"])
            return
        st.markdown(safe_markdown(msg["content"]))
        if msg["role"] != "assistant":
            return
        if msg.get("truncated"):
            st.caption("The answer hit the length limit and was cut off.")
        confidence = msg.get("confidence") if msg.get("confidence") in CONFIDENCE_COLOURS else "low"
        st.badge(f"Confidence: {confidence}", color=CONFIDENCE_COLOURS[confidence])
        sources = msg.get("sources") or []
        with st.expander(f"Sources ({len(sources)})"):
            if not sources:
                st.caption("No saved posting was close enough to use. The model wasn't asked.")
            for s in sources:
                st.markdown(f"**\\[{s['n']}\\] {plain(s['title'])}** · {plain(s['company'])} · "
                            f"distance {s['distance']}")
                st.text(s["excerpt"])


def ask_page():
    """The posting filter, the chat so far and the question box."""
    st.header("Ask about your postings")
    jobs = []
    try:
        jobs = api_client.list_jobs(st.session_state.token)
        if not jobs:
            st.info("Save a posting on the Jobs page first.")
    # not "save a posting first" when the user may well have some
    except ApiError as e:
        st.error(f"Couldn't load your postings for the filter: {e}")
    # the id keeps two postings with the same title and company apart, a dict
    # keyed on the name alone silently dropped one of them
    labels = {f"{j['title']} · {j['company']} (#{j['id']})": j["id"] for j in jobs}
    picked = st.multiselect("Only these postings (optional)", list(labels))

    for msg in st.session_state.chat:
        render(msg)

    if question := st.chat_input("e.g. Which of my saved jobs ask for Docker?", max_chars=500):
        question = question.strip()
        if question:
            user_msg = {"role": "user", "kind": "answer", "content": question}
            st.session_state.chat.append(user_msg)
            render(user_msg)
            with st.spinner("Searching your postings and asking the model..."):
                reply = {"role": "assistant", **ask(question, [labels[p] for p in picked])}
            st.session_state.chat.append(reply)
            render(reply)

    if st.session_state.chat:
        st.button("Clear chat", on_click=lambda: st.session_state.update(chat=[]))


# ── Main ───────────────────────────────────────────────────────────────────

health = api_client.health_check()
sidebar(health)

if health is None:
    st.title("JobFit")
    st.error(f"Can't reach the API at `{BACKEND_URL}`. Check `docker-compose ps backend`.")
else:
    try:
        if not st.session_state.token:
            login_page()
        else:
            page = st.sidebar.radio("Page", ["Ask", "Jobs"], label_visibility="collapsed")
            if page == "Ask":
                ask_page()
            else:
                jobs_page()
    except SessionExpired:
        # the sidebar was already drawn as signed in, so start the run over
        # and show the message on the login page instead
        logout()
        st.session_state.expired = True
        st.rerun()
    # the pages handle their own API errors, this catches any one that slips past
    except ApiError as e:
        st.error(str(e))
