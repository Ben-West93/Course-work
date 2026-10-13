"""
Every call the Streamlit UI makes to the JobFit API. app.py only uses these
functions and never calls requests itself.

Each function returns the response's JSON, or raises:

    ApiError        the API refused (status_code is its HTTP status), or no
                    usable reply came back (status_code is None). str() is
                    the message to show the user
    ApiTimeout      an ApiError for a request that ran out of time
    SessionExpired  a request that sent a token got a 401: the token ran
                    out, or the API restarted without JWT_SECRET_KEY set

    BACKEND_URL=http://localhost:8000 streamlit run app.py

This runs on the Streamlit server, not in the browser, so in Docker Compose
BACKEND_URL is the backend's service name, http://backend:8000. localhost
there would be the frontend container itself.
"""

import os

import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
# the backend gives Ollama up to 120s, so wait a bit longer than that
ASK_TIMEOUT = 150


class ApiError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ApiTimeout(ApiError):
    pass


class SessionExpired(Exception):
    pass


def error_detail(r: requests.Response) -> str:
    # app errors are {"detail": "..."}, 422s are a list of pydantic errors
    try:
        body = r.json()
    except ValueError:
        return r.text[:200] or f"HTTP {r.status_code}"
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, list) and detail:
        parts = []
        for e in detail:
            if isinstance(e, dict):
                field = ".".join(str(p) for p in e.get("loc", [])[1:])
                msg = str(e.get("msg", e)).removeprefix("Value error, ")
                parts.append(f"{field}: {msg}" if field else msg)
            else:
                parts.append(str(e))
        return "; ".join(parts)
    return str(detail or f"HTTP {r.status_code}")


def request(method: str, path: str, token: str | None = None, timeout: float = 30, **kwargs):
    """The parsed JSON of a 2xx reply, None for a 204. Raises as described above."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = requests.request(method, f"{BACKEND_URL}{path}", headers=headers, timeout=timeout, **kwargs)
    except requests.exceptions.Timeout as e:
        raise ApiTimeout(f"No answer from `{BACKEND_URL}` after {timeout:g}s.") from e
    except requests.exceptions.RequestException as e:
        raise ApiError(f"Lost the connection to `{BACKEND_URL}` ({type(e).__name__}).") from e

    # without a token a 401 is just a wrong password, with one it's a dead session
    if r.status_code == 401 and token:
        raise SessionExpired
    if r.status_code >= 400:
        raise ApiError(error_detail(r), r.status_code)
    if r.status_code == 204:
        return None
    try:
        return r.json()
    except ValueError:
        # a proxy's HTML error page, or something else answering on BACKEND_URL
        raise ApiError(f"The API's reply wasn't JSON (HTTP {r.status_code}).") from None


# ── Health and auth ────────────────────────────────────────────────────────

def health_check() -> dict | None:
    """/health's report, including the 503 it sends when the database is down. None if nothing usable answered."""
    try:
        body = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
    except (requests.exceptions.RequestException, ValueError):
        return None
    return body if isinstance(body, dict) else None


def login(username: str, password: str) -> str:
    """The access token."""
    body = request("post", "/auth/token", data={"username": username, "password": password}, timeout=10)
    return body["access_token"]


def register(username: str, password: str) -> dict:
    return request("post", "/auth/register", json={"username": username, "password": password}, timeout=10)


# ── Job postings ───────────────────────────────────────────────────────────

def list_jobs(token: str, limit: int = 100) -> list[dict]:
    return request("get", "/jobs", token, params={"limit": limit})


def create_job(token: str, posting: dict) -> dict:
    # saving also chunks and embeds it, which takes a few seconds on CPU
    return request("post", "/jobs", token, timeout=60, json=posting)


def delete_job(token: str, job_id: int) -> None:
    request("delete", f"/jobs/{job_id}", token)


def add_samples(token: str) -> dict:
    return request("post", "/jobs/samples", token, timeout=120)


def reindex(token: str) -> dict:
    return request("post", "/jobs/reindex", token, timeout=120)


# ── Ask ────────────────────────────────────────────────────────────────────

def ask(question: str, token: str, job_ids: list[int] | None = None) -> dict:
    body = {"question": question}
    if job_ids:
        body["job_ids"] = job_ids
    return request("post", "/ask", token, timeout=ASK_TIMEOUT, json=body)
