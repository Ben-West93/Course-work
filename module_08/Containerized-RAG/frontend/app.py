"""
Streamlit chat UI for the RAG assistant.

    BACKEND_URL=http://localhost:8000 streamlit run app.py

In Docker Compose BACKEND_URL is set to http://backend:8000.
"""

import os

import requests
import streamlit as st

# this code runs in the frontend container, not in the browser, so it reaches
# the backend by its compose service name. localhost would be this container
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
# the backend gives Ollama up to 120s, so wait a bit longer than that
ASK_TIMEOUT = 150
CONFIDENCE_COLOURS = {"high": "green", "medium": "orange", "low": "red"}

st.set_page_config(page_title="RAG Assistant", page_icon="🔍", layout="centered")

# Streamlit reruns this whole script on every click and message, so ordinary
# variables start over each time. session_state lasts as long as the browser
# tab, which is what keeps the conversation on screen. Keys have to exist
# before anything reads them
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ingest_result" not in st.session_state:
    st.session_state.ingest_result = None


def error_detail(r: requests.Response) -> str:
    # app errors are {"detail": "..."}, 422s are a list of pydantic errors
    try:
        body = r.json()
    except ValueError:
        return r.text[:200] or f"HTTP {r.status_code}"
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, list) and detail:
        return "; ".join(str(e.get("msg", e)).removeprefix("Value error, ") if isinstance(e, dict) else str(e)
                         for e in detail)
    return str(detail or f"HTTP {r.status_code}")


def safe_markdown(text: str) -> str:
    # st.markdown treats $...$ as LaTeX, and would load any ![image](url) in an
    # answer straight away, which a poisoned document could abuse
    return text.replace("$", "\\$").replace("![", "!\\[")


def fetch_health() -> dict | None:
    try:
        # a 503 still has the normal body (ChromaDB broken, API up)
        return requests.get(f"{BACKEND_URL}/health", timeout=5).json()
    except (requests.exceptions.RequestException, ValueError):
        return None


def reindex():
    # on_click callbacks run before the rerun, so the sidebar's /health call
    # further down already shows the new document count
    try:
        r = requests.post(f"{BACKEND_URL}/ingest", timeout=120)
    except requests.exceptions.RequestException as e:
        st.session_state.ingest_result = ("error", f"Can't reach the backend at {BACKEND_URL} ({type(e).__name__})")
        return
    if r.status_code != 200:
        st.session_state.ingest_result = ("error", f"Re-index failed ({r.status_code}): {error_detail(r)}")
        return
    data = r.json()
    kind = "success" if data.get("chunks_ingested") else "warning"
    st.session_state.ingest_result = (kind, data.get("message", "Re-indexed"))


def clear_chat():
    st.session_state.chat_history = []


def ollama_hint(detail: str) -> str:
    if detail == "Ollama unavailable":
        return "Start it with `docker-compose start ollama`."
    if detail.startswith("Ollama returned 404 for model "):
        model = detail.removeprefix("Ollama returned 404 for model ")
        return f"The model isn't pulled yet: `docker-compose exec ollama ollama pull {model}`"
    if "timed out" in detail:
        return "The model may still be loading, try again in a moment."
    return "The cause is in `docker-compose logs backend`."


def ask_backend(question: str) -> dict:
    """POST /ask and turn whatever comes back into a chat_history entry."""
    try:
        r = requests.post(f"{BACKEND_URL}/ask", json={"question": question}, timeout=ASK_TIMEOUT)
    except requests.exceptions.ConnectionError:
        return {"role": "assistant", "kind": "error",
                "content": f"Can't reach the backend at `{BACKEND_URL}`. Check `docker-compose ps`."}
    except requests.exceptions.Timeout:
        return {"role": "assistant", "kind": "error",
                "content": f"No answer after {ASK_TIMEOUT}s. The model may still be loading, try again."}
    except requests.exceptions.RequestException as e:
        return {"role": "assistant", "kind": "error", "content": f"Request failed: {type(e).__name__}"}

    if r.status_code == 422:
        return {"role": "assistant", "kind": "warning", "content": error_detail(r)}
    if r.status_code != 200:
        detail = error_detail(r)
        return {"role": "assistant", "kind": "error",
                "content": f"**{detail}** (HTTP {r.status_code}). {ollama_hint(detail)}"}
    try:
        data = r.json()
    except ValueError:
        return {"role": "assistant", "kind": "error", "content": "The backend's reply wasn't JSON."}

    return {
        "role": "assistant",
        "kind": "answer",
        "content": (data.get("answer") or "").strip() or "No answer returned.",
        "sources": data.get("sources") or [],
        "confidence": data.get("confidence"),
    }


def render_message(msg: dict):
    with st.chat_message(msg["role"]):
        kind = msg.get("kind", "answer")
        if kind != "answer":
            getattr(st, kind)(msg["content"])
            return
        st.markdown(safe_markdown(msg["content"]))
        if msg["role"] != "assistant":
            return

        confidence = msg.get("confidence")
        if confidence not in CONFIDENCE_COLOURS:
            confidence = "low"
        st.badge(f"Confidence: {confidence}", color=CONFIDENCE_COLOURS[confidence])

        sources = msg.get("sources") or []
        with st.expander(f"Sources ({len(sources)})"):
            if not sources:
                st.caption("No document was close enough to this question to use as context.")
            for s in sources:
                name = str(s.get("source") or "unknown").replace("`", "'")
                st.markdown(f"`{name}` · distance {s.get('distance', '?')}")
                st.text(s.get("text", ""))


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 RAG Assistant")
    st.caption("Answers from the course docs using ChromaDB retrieval and a local Ollama model.")

    health = fetch_health()
    if health is None:
        st.error("Backend: unreachable")
        st.caption(f"Nothing answered at `{BACKEND_URL}`.")
    else:
        st.success("Backend: connected")
        model = health.get("model", "?")
        # with Ollama down there's no way to tell whether the model is there
        if health.get("ollama") != "connected":
            pulled = "status unknown"
        else:
            pulled = "pulled" if health.get("model_pulled") else "not pulled"
        st.caption(f"ChromaDB: {health.get('chromadb', '?')} · Ollama: {health.get('ollama', '?')} · "
                   f"{model} {pulled}")
        st.metric("Documents (chunks)", health.get("document_count", 0) if health.get("chromadb") == "ok" else "?")

        if health.get("chromadb") != "ok":
            st.error("ChromaDB can't be read. Check `docker-compose logs backend`.")
        elif health.get("ollama") != "connected":
            st.warning("Ollama isn't answering, so questions will fail. `docker-compose start ollama`")
        elif not health.get("model_pulled"):
            st.warning(f"Pull the model first: `docker-compose exec ollama ollama pull {model}`")
        elif not health.get("document_count"):
            st.info("Nothing indexed yet. Click Re-index Documents.")

    st.divider()
    st.button("Re-index Documents", on_click=reindex, disabled=health is None, width="stretch")
    if st.session_state.ingest_result:
        kind, message = st.session_state.ingest_result
        getattr(st, kind)(message)
    st.button("Clear chat", on_click=clear_chat, width="stretch")


# ── Main view ──────────────────────────────────────────────────────────────
st.title("Ask the docs")
st.caption("Questions about Python, FastAPI, Streamlit, Docker or RAG, answered from the indexed documents.")

for msg in st.session_state.chat_history:
    render_message(msg)

if question := st.chat_input("Ask a question...", max_chars=1000):
    question = question.strip()
    if question:
        user_msg = {"role": "user", "content": question}
        st.session_state.chat_history.append(user_msg)
        render_message(user_msg)
        with st.spinner("Searching the docs and asking the model..."):
            reply = ask_backend(question)
        st.session_state.chat_history.append(reply)
        render_message(reply)
