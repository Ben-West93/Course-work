"""
Frontend: Streamlit RAG UI
Run with (local dev):
    BACKEND_URL=http://localhost:8000 streamlit run app.py

Inside Docker it reads BACKEND_URL from the environment, docker-compose.yml
sets it to http://backend:8000.
"""

import os

import requests
import streamlit as st

# this code runs in the frontend container, not the browser, so it reaches the
# backend by its compose service name. localhost would be this container
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
# the backend gives Ollama up to 120s, so wait a bit longer than that
ASK_TIMEOUT = 150
COLOURS = {"high": "green", "medium": "orange", "low": "red"}
MODEL_HINT = "docker-compose exec ollama ollama pull llama3.2:1b"

st.set_page_config(page_title="RAG Assistant", page_icon="🔍", layout="centered")

# Streamlit reruns this whole file on every click and submit, so plain
# variables are gone by the next run. session_state lasts as long as the
# browser tab, which is what keeps the last answer on screen after clicking
# Re-index, and the last re-index result after asking
st.session_state.setdefault("result", None)
st.session_state.setdefault("ingest", None)


def error_detail(r: requests.Response) -> str:
    # app errors are {"detail": "..."}, 422s are a list of pydantic errors
    try:
        body = r.json()
    except ValueError:
        return r.text[:200] or f"HTTP {r.status_code}"
    detail = body.get("detail") if isinstance(body, dict) else body
    if isinstance(detail, list) and detail:
        return "; ".join(str(e.get("msg", e)).removeprefix("Value error, ") if isinstance(e, dict) else str(e)
                         for e in detail)
    return str(detail or f"HTTP {r.status_code}")


def fetch_health() -> dict | None:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        # a 503 here still has the usual body, ChromaDB is broken but the API is up
        return r.json()
    except (requests.exceptions.RequestException, ValueError):
        return None


def reindex():
    # on_click callbacks run before the rerun the click triggers, so the
    # sidebar's /health call below already sees the new chunk count
    try:
        r = requests.post(f"{BACKEND_URL}/ingest", timeout=120)
    except requests.exceptions.ConnectionError:
        st.session_state.ingest = ("error", f"Can't reach the backend at `{BACKEND_URL}`.")
        return
    except requests.exceptions.RequestException as e:
        st.session_state.ingest = ("error", f"Re-index failed: {type(e).__name__}")
        return

    if r.status_code != 200:
        st.session_state.ingest = ("error", f"Re-index failed ({r.status_code}): {error_detail(r)}")
        return
    data = r.json()
    # an empty docs/ folder is a 200 with zero counts and nothing changed
    kind = "success" if data.get("chunks_ingested") else "warning"
    st.session_state.ingest = (kind, data.get("message", "Re-indexed"))


def ollama_hint(detail: str) -> str:
    if detail == "Ollama unavailable":
        return "Ollama isn't running. Start it with `docker-compose start ollama` and ask again."
    if "for model" in detail:
        return f"The model isn't pulled yet: `{MODEL_HINT}`"
    if "timed out" in detail:
        return "The model may still be loading. Give it a moment and ask again."
    return "Check `docker-compose logs backend` for the cause."


def ask(question: str) -> dict:
    try:
        r = requests.post(f"{BACKEND_URL}/ask", json={"question": question}, timeout=ASK_TIMEOUT)
    except requests.exceptions.ConnectionError:
        return {"kind": "error", "message": f"Can't reach the backend at `{BACKEND_URL}`. Check `docker-compose ps` "
                                            "and start it with `docker-compose up -d backend`."}
    except requests.exceptions.Timeout:
        return {"kind": "error", "message": f"No answer after {ASK_TIMEOUT}s. The model may still be loading, "
                                            "try again."}
    except requests.exceptions.RequestException as e:
        return {"kind": "error", "message": f"Request to the backend failed: {type(e).__name__}"}

    if r.status_code == 422:
        return {"kind": "warning", "message": error_detail(r)}
    if r.status_code in (502, 503):
        detail = error_detail(r)
        return {"kind": "error", "message": f"**{detail}** (HTTP {r.status_code}). {ollama_hint(detail)}"}
    if r.status_code != 200:
        return {"kind": "error", "message": f"Backend returned {r.status_code}: {error_detail(r)}"}
    try:
        return {"kind": "answer", "question": question, "data": r.json()}
    except ValueError:
        return {"kind": "error", "message": "The backend's reply wasn't JSON."}


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    health = fetch_health()
    if health is None:
        st.error("Backend: unreachable ✗")
        st.caption(f"Nothing answered at `{BACKEND_URL}`. Start it with `docker-compose up -d backend`.")
    else:
        st.success("Backend: connected ✓")
        count = health.get("document_count")
        st.metric("Chunks stored", "?" if count is None else count)
        # with Ollama down there's no way to tell whether the model is pulled
        if health.get("ollama") == "connected":
            pulled = "pulled" if health.get("model_pulled") else "not pulled"
            st.caption(f"Ollama: connected · {health.get('model', '?')} {pulled}")
        else:
            st.caption("Ollama: disconnected")

        if health.get("chromadb") == "error":
            st.error("ChromaDB can't be read. Check `docker-compose logs backend`.")
        elif health.get("ollama") != "connected":
            st.warning("Ollama isn't answering, so Ask will fail. `docker-compose start ollama`")
        elif not health.get("model_pulled"):
            st.warning(f"Pull the model before asking: `{MODEL_HINT}`")
        elif count == 0:
            st.info("Nothing ingested yet. Click Re-index Documents.")

    st.button("Re-index Documents", on_click=reindex, disabled=health is None, width="stretch")
    if st.session_state.ingest:
        kind, message = st.session_state.ingest
        getattr(st, kind)(message)

# ── Main area ──────────────────────────────────────────────────────────────
st.title("RAG Assistant")
st.caption("Ask questions grounded in your documents.")

# a form only reruns the script when it's submitted, not on every keystroke,
# and Enter submits it
with st.form("ask_form"):
    question = st.text_input("Your question", max_chars=1000, placeholder="How does ChromaDB store embeddings?")
    submitted = st.form_submit_button("Ask", type="primary")

if submitted:
    if not question.strip():
        st.session_state.result = {"kind": "warning", "message": "Type a question first."}
    else:
        with st.spinner("Searching the docs and asking the model..."):
            st.session_state.result = ask(question.strip())

result = st.session_state.result
if result and result["kind"] == "answer":
    data = result["data"]
    answer = (data.get("answer") or "").strip()
    confidence = data.get("confidence") if data.get("confidence") in COLOURS else "low"
    sources = data.get("sources") or []

    st.subheader("Answer")
    st.badge(f"Confidence: {confidence}", color=COLOURS[confidence])
    if answer:
        st.markdown(answer)
    else:
        st.info("No answer returned")
    if confidence == "low" and sources:
        st.caption("Low confidence: the closest source is only loosely related, or two documents match "
                   "about equally well. Check the sources before relying on this.")

    with st.expander(f"Sources ({len(sources)})"):
        if not sources:
            st.caption("No document was close enough to this question to use as context.")
        for s in sources:
            st.markdown(f"**{s.get('source', 'unknown')}** · distance {s.get('distance', '?')}")
            st.text(s.get("text", ""))
elif result:
    getattr(st, result["kind"])(result["message"])
