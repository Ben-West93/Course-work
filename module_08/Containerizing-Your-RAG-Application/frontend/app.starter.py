"""
Frontend: Streamlit RAG UI  (STARTER)
=======================================
Run with (local dev):
    BACKEND_URL=http://localhost:8000 streamlit run app.py

Inside Docker it reads BACKEND_URL from the environment automatically.

Your goal: build a Streamlit UI that talks to the FastAPI RAG backend.

Required features:
    1. Sidebar:
       - Health status indicator (call GET /health on load)
         Show "Backend: connected ✓" or "Backend: unreachable ✗"
       - "Re-index Documents" button (calls POST /ingest)
         Show success message with chunk count
    2. Main area:
       - Page title and brief description
       - Text input for the question
       - "Ask" button (or use st.chat_input)
       - Answer display with:
           - Confidence badge (high / medium / low)
           - Answer text
           - Source list (document names)
    3. Error handling:
       - Backend unreachable → show st.error() with instructions
       - Empty answer → show a "No answer returned" message

Key concepts:
    Read backend URL from env:
        import os
        BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

    Calling the backend:
        response = requests.post(f"{BACKEND_URL}/ask",
                                 json={"question": question})
        data = response.json()
        # data["answer"], data["sources"], data["confidence"]

    Confidence colours:
        colours = {"high": "green", "medium": "orange", "low": "red"}
        st.markdown(f":{colour}[Confidence: {confidence}]")
"""

import streamlit as st
import requests
import os

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# ── Page config ────────────────────────────────────────────────────────────
# TODO: st.set_page_config(page_title="RAG Assistant", page_icon="🔍", layout="centered")

# ── Sidebar ────────────────────────────────────────────────────────────────
# TODO: with st.sidebar:
#   - st.header("Controls")
#   - Fetch GET /health and show backend status
#   - "Re-index Documents" button → POST /ingest → show result

# ── Main area ──────────────────────────────────────────────────────────────
# TODO: st.title("RAG Assistant")
# TODO: st.caption("Ask questions grounded in your documents.")

# TODO: question = st.text_input("Your question") or st.chat_input(...)

# TODO: if question:
#   - POST /ask with {"question": question}
#   - Display confidence badge
#   - Display answer
#   - Display sources (st.expander or bullet list)
#   - Handle requests.exceptions.ConnectionError with st.error()
