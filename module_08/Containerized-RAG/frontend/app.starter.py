"""
Module 8 Project — Containerized RAG Assistant  (STARTER)
Frontend: Streamlit Chat UI
============================
Run locally:
    BACKEND_URL=http://localhost:8000 streamlit run app.py

Inside Docker Compose: started automatically; reads BACKEND_URL from env.

Your task: implement the four sections below.

Required features:
    1. Session state    — initialise chat_history list before any reads
    2. Sidebar          — health check, document count, Re-index button
    3. Chat display     — render existing messages with st.chat_message()
    4. Chat input       — st.chat_input(), call /ask, append to history

API endpoints to call (all relative to BACKEND_URL):
    GET  /health  — check service status
    POST /ingest  — trigger document re-indexing
    POST /ask     — send question, get {answer, sources, confidence}
"""

import streamlit as st
import requests
import os

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# ── Step 1: Page configuration ─────────────────────────────────────────────
# TODO: st.set_page_config(page_title="RAG Assistant", page_icon="🔍", layout="centered")

# ── Step 2: Session state initialisation ───────────────────────────────────
# TODO: Initialise "chat_history" as an empty list if it doesn't exist yet.
# Always initialise session state keys before reading them to avoid KeyError.
#
#   if "chat_history" not in st.session_state:
#       st.session_state.chat_history = []


# ════════════════════════════════════════════════════════════════════════════
# SECTION A — SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
# TODO: with st.sidebar:
#   - App title and a brief description
#   - Health check: GET BACKEND_URL + "/health"
#     Show st.success("Backend: connected") or st.error("Backend: unreachable")
#     Show a caption with chromadb status, ollama status, and document count
#   - st.divider()
#   - "Re-index Documents" button:
#     On click: POST BACKEND_URL + "/ingest"
#     Show st.success with the number of chunks ingested, or st.error on failure


# ════════════════════════════════════════════════════════════════════════════
# SECTION B — MAIN CONTENT
# ════════════════════════════════════════════════════════════════════════════
# TODO: st.title("RAG Assistant")
# TODO: st.caption("Ask questions grounded in your documents.")

# ── Chat history display ───────────────────────────────────────────────────
# TODO: Loop over st.session_state.chat_history and render each message.
#
# Each message in history is a dict: {"role": "user"|"assistant", "content": str}
# Use st.chat_message(role) as a context manager:
#
#   for msg in st.session_state.chat_history:
#       with st.chat_message(msg["role"]):
#           st.markdown(msg["content"])
#
# For assistant messages that have a "sources" key, show citations in an expander.
# For assistant messages that have a "confidence" key, show a colour-coded badge.


# ── Chat input ─────────────────────────────────────────────────────────────
# TODO: Use st.chat_input("Ask a question...") to get user input.
# When the user submits a question:
#   1. Append {"role": "user", "content": question} to chat_history
#   2. Display the user message immediately with st.chat_message("user")
#   3. Show a spinner while calling POST BACKEND_URL + "/ask"
#      Payload: {"question": question}
#   4. On success:
#      - Display the answer with st.chat_message("assistant")
#      - Show confidence badge (green=high, orange=medium, red=low)
#      - Show source citations in an st.expander
#      - Append {"role": "assistant", "content": answer, "sources": [...], "confidence": "..."} to chat_history
#   5. On requests.exceptions.ConnectionError:
#      - Show st.error("Cannot reach the backend at ...")
#   6. On HTTP error (e.g. 503 Ollama not running):
#      - Show st.error with the error detail from the response

# ── Hint: colour-coded confidence badge ───────────────────────────────────
# CONFIDENCE_COLOURS = {"high": "green", "medium": "orange", "low": "red"}
# colour = CONFIDENCE_COLOURS.get(confidence, "gray")
# st.markdown(f":{colour}[Confidence: **{confidence}**]")
