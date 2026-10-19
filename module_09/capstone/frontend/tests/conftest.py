"""
The fake API shared by test_app.py and test_api_client.py. No backend needed.

`fake_api` swaps requests.request/get/post for a router keyed on
(method, path), which is all api_client.py uses. Each test sets the API's
replies, then checks what the client returned or the UI showed.
"""

import json

import pytest
import requests

HEALTH = {
    "status": "ok",
    "database": "ok",
    "chroma": "ok",
    "ollama": "connected",
    "model_ready": True,
    "model": "llama3.2:3b",
}
JOB = {"id": 1, "title": "Dev", "company": "Same Co", "location": None, "source": "pasted", "indexed": True}


class Reply:
    """A stand-in for requests.Response: a status, a JSON body and its text."""

    def __init__(self, status_code=200, body=None, text=None):
        """`body` is what json() returns, None for a reply that isn't JSON. `text` defaults to the body as JSON."""
        self.status_code, self._body = status_code, body
        self.text = text if text is not None else json.dumps(body)

    def json(self):
        """The body, or a ValueError like requests raises when it isn't JSON."""
        if self._body is None:
            raise ValueError("not json")
        return self._body


@pytest.fixture
def fake_api(monkeypatch):
    """
    Returns the routes dict. A value is a Reply, an exception to raise, or a
    callable that gets the request's keyword arguments (json=, data=,
    headers=, timeout=, ...).
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
