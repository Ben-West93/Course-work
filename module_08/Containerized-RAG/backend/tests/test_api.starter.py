"""
Module 8 Project — Containerized RAG Assistant  (STARTER)
Backend: API Tests
===================
Run with:
    pytest tests/ -v

At least 3 tests are required for full marks.
These smoke tests verify your API returns the right HTTP status codes and
response shapes — they don't require a running Ollama or pre-loaded data.

TODO: Implement all three test functions below.
"""

from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: E402

client = TestClient(app)


def test_root():
    """
    GET / should return HTTP 200 with a "message" field.

    TODO: Call client.get("/") and assert:
      - response.status_code == 200
      - "message" in response.json()
    """
    # TODO: implement
    pass


def test_health():
    """
    GET /health should return HTTP 200 with status, chromadb, and ollama keys.

    TODO: Call client.get("/health") and assert:
      - response.status_code == 200
      - "status"   in response.json()
      - "chromadb" in response.json()
      - "ollama"   in response.json()
    """
    # TODO: implement
    pass


def test_stats():
    """
    GET /stats should return HTTP 200 with document_count and model keys.

    TODO: Call client.get("/stats") and assert:
      - response.status_code == 200
      - "document_count" in response.json()
      - "model"          in response.json()
    """
    # TODO: implement
    pass
