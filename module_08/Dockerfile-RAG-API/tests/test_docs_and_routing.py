import pytest
import requests

import my_rag_api as api


@pytest.fixture
def spec(client):
    return client.get("/openapi.json").json()


# ── swagger / openapi ──────────────────────────────────────────────────────

def test_docs_pages_load(client):
    assert "swagger-ui" in client.get("/docs").text
    assert "redoc" in client.get("/redoc").text.lower()


def test_every_route_is_listed(spec):
    assert {p: sorted(ops) for p, ops in spec["paths"].items()} == {
        "/ask": ["post"], "/ingest": ["post"], "/stats": ["get"], "/health": ["get"],
    }


def test_every_route_has_a_summary_and_description(spec):
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            assert op.get("summary"), f"{method} {path}"
            assert op.get("description"), f"{method} {path}"


@pytest.mark.parametrize("path, method, codes", [
    ("/ask", "post", {"200", "400", "422", "502", "503"}),
    ("/ingest", "post", {"200"}),
    ("/stats", "get", {"200", "503"}),
    ("/health", "get", {"200", "503"}),
])
def test_error_responses_are_documented(spec, path, method, codes):
    assert set(spec["paths"][path][method]["responses"]) == codes


def test_every_schema_field_has_a_description(spec):
    ours = ["AskRequest", "AskResponse", "SourceChunk", "IngestResponse",
            "StatsResponse", "HealthResponse", "ErrorResponse"]
    for name in ours:
        for field, prop in spec["components"]["schemas"][name]["properties"].items():
            assert prop.get("description"), f"{name}.{field}"


def test_request_limits_are_in_the_schema(spec):
    props = spec["components"]["schemas"]["AskRequest"]["properties"]
    assert spec["components"]["schemas"]["AskRequest"]["required"] == ["question"]
    assert (props["question"]["minLength"], props["question"]["maxLength"]) == (1, 1000)
    assert (props["n_results"]["minimum"], props["n_results"]["maximum"], props["n_results"]["default"]) == (1, 20, 3)
    max_distance = props["max_distance"]
    assert (max_distance["minimum"], max_distance["maximum"], max_distance["default"]) == (0, 4, 1.2)


def test_fixed_value_fields_are_enums(spec):
    schemas = spec["components"]["schemas"]
    assert schemas["AskResponse"]["properties"]["confidence"]["enum"] == ["high", "medium", "low"]
    assert schemas["HealthResponse"]["properties"]["status"]["enum"] == ["ok", "degraded", "error"]


def test_examples_are_valid_against_the_models(spec):
    schemas = spec["components"]["schemas"]
    for example in schemas["AskRequest"]["examples"]:
        api.AskRequest(**example)
    for example in schemas["AskResponse"]["examples"]:
        api.AskResponse(**example)


# ── routing ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("method, path", [
    ("GET", "/ask"), ("PUT", "/ask"), ("GET", "/ingest"), ("DELETE", "/ingest"),
    ("POST", "/stats"), ("DELETE", "/stats"), ("POST", "/health"),
])
def test_wrong_method_is_405(client, method, path):
    assert client.request(method, path).status_code == 405


@pytest.mark.parametrize("path", ["/", "/nope", "/ask/extra", "/api/ask"])
def test_unknown_path_is_404(client, path):
    assert client.get(path).status_code == 404


def test_cors_preflight(client):
    r = client.options("/ask", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_header_on_normal_requests(client, empty_collection):
    r = client.get("/health", headers={"Origin": "http://example.com"})
    assert r.headers["access-control-allow-origin"] == "http://example.com"


def documented_example(spec, path, method, code):
    return spec["paths"][path][method]["responses"][code]["content"]["application/json"]["example"]


class BrokenCollection:
    def count(self):
        raise RuntimeError("database is locked")


def test_documented_error_examples_match_real_responses(client, ingested, ollama, spec, monkeypatch):
    ollama.chat_error = requests.exceptions.ConnectionError()
    assert client.post("/ask", json={"question": "What is chunking?"}).json() == \
        documented_example(spec, "/ask", "post", "503")

    ollama.chat_error = None
    ollama.reply(200, {"message": {"content": ""}})
    assert client.post("/ask", json={"question": "What is chunking?"}).json() == \
        documented_example(spec, "/ask", "post", "502")

    r = client.post("/ask", content=b'{"question": "\xff"}', headers={"Content-Type": "application/json"})
    assert r.json() == documented_example(spec, "/ask", "post", "400")

    monkeypatch.setattr(api, "collection", BrokenCollection())
    assert client.get("/stats").json() == documented_example(spec, "/stats", "get", "503")

    health = client.get("/health").json()
    assert health["document_count"] is None
    assert {k: v for k, v in health.items() if v is not None} == documented_example(spec, "/health", "get", "503")
