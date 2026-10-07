import pytest
import requests

import my_rag_api as api
from conftest import BrokenCollection


def ask(client, question="What is chunking?", **extra):
    return client.post("/ask", json={"question": question, **extra})


# ── valid requests ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("question, expected_source", [
    ("How does ChromaDB store documents in collections?", "chromadb.md"),
    ("Why split long documents into smaller pieces before embedding?", "chunking.txt"),
    ("What does an embedding vector represent?", "embeddings.md"),
    ("How do I start the Ollama server and pull a model?", "ollama.txt"),
    ("What is prompt injection in a RAG system?", "prompt_injection.md"),
    ("How does RAG reduce hallucination?", "rag_overview.md"),
    ("How does semantic search rank results by meaning?", "semantic_search.md"),
])
def test_question_retrieves_the_right_doc(client, ingested, ollama, question, expected_source):
    r = ask(client, question)
    assert r.status_code == 200
    assert r.json()["sources"][0]["source"] == expected_source
    assert len(ollama.calls) == 1


def test_response_fields_are_populated_and_consistent(client, ingested, ollama):
    r = ask(client, "How does ChromaDB store embeddings?")
    body = r.json()

    assert set(body) == {"answer", "sources", "confidence"}
    assert body["answer"] == ollama.answer
    assert body["confidence"] in {"high", "medium", "low"}
    assert 1 <= len(body["sources"]) <= 3

    distances = [s["distance"] for s in body["sources"]]
    assert distances == sorted(distances)
    for s in body["sources"]:
        assert set(s) == {"text", "source", "distance"}
        assert s["text"]
        assert s["text"].strip() == s["text"]
        assert s["source"].endswith((".md", ".txt"))
        assert 0 <= s["distance"] <= api.settings.confidence_threshold
        assert round(s["distance"], 4) == s["distance"]


def test_prompt_sent_to_ollama(client, ingested, ollama):
    r = ask(client, "  How does ChromaDB store embeddings?  ")
    sources = r.json()["sources"]
    call = ollama.calls[0]
    system, user = call["json"]["messages"]

    assert call["url"] == f"{api.settings.ollama_url}/api/chat"
    assert call["json"]["model"] == api.settings.model_name
    assert call["json"]["stream"] is False
    assert call["json"]["options"]["temperature"] == 0
    assert user == {"role": "user", "content": "How does ChromaDB store embeddings?"}
    assert system["role"] == "system"
    assert system["content"].startswith(api.SYSTEM_PROMPT)
    for s in sources:
        assert f"[Source: {s['source']}]" in system["content"]
        assert s["text"] in system["content"]


def test_n_results_limits_sources(client, ingested):
    assert len(ask(client, "What is ChromaDB?", n_results=1).json()["sources"]) == 1
    assert len(ask(client, "What is ChromaDB?", n_results=20, max_distance=4).json()["sources"]) == 20


def test_n_results_bigger_than_collection(client, empty_collection):
    empty_collection.add(ids=["a", "b"], documents=["ChromaDB stores vectors.", "Ollama runs models."],
                         metadatas=[{"source": "a.md"}, {"source": "b.md"}])
    r = ask(client, "What is ChromaDB?", n_results=20, max_distance=4)
    assert r.status_code == 200
    assert len(r.json()["sources"]) == 2


@pytest.mark.parametrize("max_distance", [0.6, 0.9, 1.2])
def test_max_distance_filters_sources(client, ingested, max_distance):
    r = ask(client, "How does ChromaDB store embeddings?", n_results=10, max_distance=max_distance)
    assert all(s["distance"] <= max_distance for s in r.json()["sources"])


@pytest.mark.parametrize("payload", [
    {"question": "x"},
    {"question": "a" * 1000},
    {"question": "What is ChromaDB?", "n_results": 1, "max_distance": 0},
    {"question": "What is ChromaDB?", "n_results": 20, "max_distance": 4},
    {"question": "¿Qué es ChromaDB? 日本語 🚀"},
    {"question": "What is ChromaDB?", "unexpected": "ignored"},
])
def test_boundary_and_unusual_payloads_accepted(client, ingested, payload):
    r = client.post("/ask", json=payload)
    assert r.status_code == 200
    assert r.json()["confidence"] in {"high", "medium", "low"}


# ── nothing to answer from ─────────────────────────────────────────────────

def test_empty_collection_skips_the_model(client, empty_collection, ollama):
    r = ask(client)
    assert r.status_code == 200
    assert r.json() == {"answer": api.EMPTY_DB_ANSWER, "sources": [], "confidence": "low"}
    assert ollama.calls == []


def test_nothing_within_max_distance_skips_the_model(client, ingested, ollama):
    r = ask(client, max_distance=0)
    assert r.json() == {"answer": api.NO_MATCH_ANSWER, "sources": [], "confidence": "low"}
    assert ollama.calls == []


@pytest.mark.parametrize("question", [
    "Who won the 1998 FIFA World Cup?",
    "Give me a pancake recipe",
    "asdf qwer zxcv",
    "How does it work?",
    "Tell me more",
])
def test_off_topic_and_vague_questions_get_no_sources(client, ingested, ollama, question):
    r = ask(client, question)
    assert r.json()["answer"] == api.NO_MATCH_ANSWER
    assert r.json()["sources"] == []
    assert ollama.calls == []


def test_no_match_still_works_with_ollama_down(client, ingested, ollama):
    ollama.chat_error = requests.exceptions.ConnectionError()
    assert ask(client, max_distance=0).status_code == 200


# ── invalid payloads ───────────────────────────────────────────────────────

@pytest.mark.parametrize("payload, field", [
    ({"question": ""}, "question"),
    ({"question": "   \n\t"}, "question"),
    ({"question": "a" * 1001}, "question"),
    ({}, "question"),
    ({"question": None}, "question"),
    ({"question": 123}, "question"),
    ({"question": ["What is RAG?"]}, "question"),
    ({"question": "hi", "n_results": 0}, "n_results"),
    ({"question": "hi", "n_results": -1}, "n_results"),
    ({"question": "hi", "n_results": 21}, "n_results"),
    ({"question": "hi", "n_results": 2.5}, "n_results"),
    ({"question": "hi", "n_results": "abc"}, "n_results"),
    ({"question": "hi", "n_results": None}, "n_results"),
    ({"question": "hi", "n_results": True}, "n_results"),
    ({"question": "hi", "n_results": "5"}, "n_results"),
    ({"question": "hi", "n_results": 5.0}, "n_results"),
    ({"question": "hi", "max_distance": -0.1}, "max_distance"),
    ({"question": "hi", "max_distance": 4.01}, "max_distance"),
    ({"question": "hi", "max_distance": "far"}, "max_distance"),
    ({"question": "hi", "max_distance": "0.9"}, "max_distance"),
    ({"question": "hi", "max_distance": True}, "max_distance"),
])
def test_invalid_fields_rejected(client, ingested, ollama, payload, field):
    r = client.post("/ask", json=payload)
    assert r.status_code == 422
    assert field in [e["loc"][-1] for e in r.json()["detail"]]
    assert ollama.calls == []


@pytest.mark.parametrize("content, headers", [
    (b'{"question": "broken json"', {"Content-Type": "application/json"}),
    (b"", {"Content-Type": "application/json"}),
    (b'["What is RAG?"]', {"Content-Type": "application/json"}),
    (b"question=hi", {"Content-Type": "application/x-www-form-urlencoded"}),
    (b"What is RAG?", {"Content-Type": "text/plain"}),
])
def test_malformed_bodies_rejected(client, ollama, content, headers):
    r = client.post("/ask", content=content, headers=headers)
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], list)
    assert ollama.calls == []


@pytest.mark.parametrize("content, field, error_type", [
    pytest.param(b'{"question": "hi", "max_distance": NaN}', "max_distance", "finite_number", id="nan"),
    pytest.param(b'{"question": "hi", "max_distance": Infinity}', "max_distance", "finite_number", id="infinity"),
    pytest.param(b'{"question": "hi", "max_distance": -Infinity}', "max_distance", "finite_number",
                 id="minus_infinity"),
    pytest.param(b'{"question": "What is \\ud800 chunking?"}', "question", "string_unicode", id="lone_surrogate"),
])
def test_values_that_cant_be_echoed_back_still_give_422(client, ollama, content, field, error_type):
    # these used to crash the 422 response itself and come back as a 500
    r = client.post("/ask", content=content, headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    error = r.json()["detail"][0]
    assert (error["loc"][-1], error["type"]) == (field, error_type)
    assert ollama.calls == []


# 5,000 levels is enough on Python 3.11 but 3.13's parser copes with it,
# so go well past both
@pytest.mark.parametrize("content", [
    pytest.param(b'{"question": "What is \xff chunking?"}', id="invalid_utf8"),
    pytest.param(('{"question": "hi", "x": ' + "[" * 50_000 + "]" * 50_000 + "}").encode(), id="nested_50k"),
])
def test_unreadable_body_is_400(client, ollama, content):
    r = client.post("/ask", content=content, headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert r.json() == {"detail": "There was an error parsing the body"}
    assert ollama.calls == []


# ── Ollama failures ────────────────────────────────────────────────────────

@pytest.mark.parametrize("error, detail", [
    (requests.exceptions.ConnectionError(), "Ollama service is unavailable"),
    (requests.exceptions.Timeout(), "Ollama timed out"),
    (requests.exceptions.InvalidURL(), "Could not reach Ollama (InvalidURL)"),
])
def test_ollama_unreachable_is_503(client, ingested, ollama, error, detail):
    ollama.chat_error = error
    r = ask(client)
    assert r.status_code == 503
    assert r.json() == {"detail": detail}


@pytest.mark.parametrize("status", [404, 500])
def test_ollama_error_status_is_503(client, ingested, ollama, status):
    ollama.reply(status, {"error": "model 'llama3.2:1b' not found"})
    r = ask(client)
    assert r.status_code == 503
    assert r.json() == {"detail": f"Ollama returned {status} for model {api.settings.model_name}"}


@pytest.mark.parametrize("data, bad_json, detail", [
    pytest.param(None, True, "Ollama's response had no answer in it", id="not_json"),
    pytest.param({"error": "something"}, False, "Ollama's response had no answer in it", id="no_message"),
    pytest.param({"message": None}, False, "Ollama's response had no answer in it", id="message_null"),
    pytest.param({"message": {"role": "assistant"}}, False, "Ollama's response had no answer in it", id="no_content"),
    pytest.param({"message": {"content": ""}}, False, "Ollama returned an empty answer", id="empty"),
    pytest.param({"message": {"content": "  \n "}}, False, "Ollama returned an empty answer", id="whitespace"),
    pytest.param({"message": {"content": None}}, False, "Ollama returned an empty answer", id="content_null"),
    pytest.param({"message": {"content": 42}}, False, "Ollama returned an empty answer", id="content_number"),
])
def test_unusable_ollama_reply_is_502(client, ingested, ollama, data, bad_json, detail):
    ollama.reply(200, data, bad_json)
    r = ask(client)
    assert r.status_code == 502
    assert r.json() == {"detail": detail}


@pytest.mark.parametrize("error", [
    requests.exceptions.ChunkedEncodingError(),
    requests.exceptions.ContentDecodingError(),
])
def test_cut_off_ollama_reply_is_502(client, ingested, ollama, error):
    ollama.chat_error = error
    r = ask(client)
    assert r.status_code == 502
    assert r.json() == {"detail": "Ollama's response was cut off"}


def test_ollama_failures_are_logged_with_the_cause(client, ingested, ollama, caplog):
    ollama.reply(404, {"error": "model \"llama3.2:1b\" not found, try pulling it first"})
    ask(client)
    assert "returning 503" in caplog.text
    assert "try pulling it first" in caplog.text


def test_answer_whitespace_is_trimmed(client, ingested, ollama):
    ollama.answer = "\n  Chunking splits documents.  \n"
    assert ask(client).json()["answer"] == "Chunking splits documents."


# ── odd data in the collection ─────────────────────────────────────────────

def test_chunk_without_source_is_labelled_unknown(client, empty_collection):
    empty_collection.add(ids=["no_meta"], documents=["ChromaDB is a vector database."])
    r = ask(client, "What is ChromaDB?")
    assert r.status_code == 200
    assert r.json()["sources"][0]["source"] == "unknown"


def test_chunk_with_empty_source_is_labelled_unknown(client, empty_collection):
    empty_collection.add(ids=["blank_source"], documents=["ChromaDB is a vector database."],
                         metadatas=[{"source": ""}])
    assert ask(client, "What is ChromaDB?").json()["sources"][0]["source"] == "unknown"


def test_blank_chunks_are_never_returned(client, empty_collection):
    empty_collection.add(
        ids=["blank", "real"],
        documents=["   ", "ChromaDB is a vector database."],
        metadatas=[{"source": "blank.md"}, {"source": "real.md"}],
    )
    sources = ask(client, "What is ChromaDB?", n_results=2, max_distance=4).json()["sources"]
    assert [s["source"] for s in sources] == ["real.md"]


def test_only_blank_chunks_gives_no_match(client, empty_collection, ollama):
    empty_collection.add(ids=["blank"], documents=[" "], metadatas=[{"source": "blank.md"}])
    r = ask(client, "What is ChromaDB?", max_distance=4)
    assert r.json() == {"answer": api.NO_MATCH_ANSWER, "sources": [], "confidence": "low"}
    assert ollama.calls == []


def test_retrieve_on_empty_collection(empty_collection):
    # chroma raises if asked for 0 results, so retrieve has to bail out first
    assert api.retrieve("anything") == []


@pytest.mark.parametrize("value, expected", [
    (123, "123"),
    (4.5, "4.5"),
    (True, "unknown"),
    (False, "unknown"),
    ("   ", "unknown"),
    ("\n\t", "unknown"),
    (" chromadb.md ", "chromadb.md"),
    ("docs/chromadb.md", "docs/chromadb.md"),
])
def test_odd_source_values_are_cleaned_up(client, empty_collection, value, expected):
    # numbers and bools used to fail response validation and come back as a 500
    empty_collection.add(ids=["a"], documents=["ChromaDB is a vector database."], metadatas=[{"source": value}])
    r = ask(client, "What is ChromaDB?")
    assert r.status_code == 200
    assert r.json()["sources"][0]["source"] == expected


def test_metadata_without_a_source_key(client, empty_collection):
    empty_collection.add(ids=["a"], documents=["ChromaDB is a vector database."], metadatas=[{"chunk_index": 0}])
    assert ask(client, "What is ChromaDB?").json()["sources"][0]["source"] == "unknown"


def test_duplicate_text_is_returned_and_sent_once(client, empty_collection, ollama):
    empty_collection.add(
        ids=["a", "b", "c"],
        documents=["ChromaDB is a vector database.", "  ChromaDB is a vector database.\n", "Ollama runs models."],
        metadatas=[{"source": "a.md"}, {"source": "copy.md"}, {"source": "b.md"}],
    )
    sources = ask(client, "What is ChromaDB?", n_results=3, max_distance=4).json()["sources"]
    assert [s["text"] for s in sources] == ["ChromaDB is a vector database.", "Ollama runs models."]
    system = ollama.calls[0]["json"]["messages"][0]["content"]
    assert system.count("ChromaDB is a vector database.") == 1


def test_same_text_from_two_files_keeps_the_closer_one(client, scripted):
    scripted(("Shared footer.", {"source": "a.md"}, 0.4), ("Shared footer.", {"source": "b.md"}, 0.6),
             ("Other text.", {"source": "c.md"}, 0.7))
    sources = ask(client, n_results=3).json()["sources"]
    assert [(s["source"], s["distance"]) for s in sources] == [("a.md", 0.4), ("c.md", 0.7)]


# ── exact distances (scripted collection) ──────────────────────────────────

def test_chunk_exactly_at_max_distance_is_kept(client, scripted):
    scripted(("at the limit", {"source": "a.md"}, 0.9), ("just past it", {"source": "b.md"}, 0.90001))
    sources = ask(client, n_results=2, max_distance=0.9).json()["sources"]
    assert [s["source"] for s in sources] == ["a.md"]


def test_distance_is_rounded_to_4_places(client, scripted):
    scripted(("text", {"source": "a.md"}, 0.123456789))
    assert ask(client).json()["sources"][0]["distance"] == 0.1235


def test_none_text_and_none_metadata(client, scripted):
    scripted((None, {"source": "empty.md"}, 0.2), ("real text", None, 0.3))
    assert ask(client, n_results=2).json()["sources"] == [{"text": "real text", "source": "unknown", "distance": 0.3}]


def test_only_blank_or_too_far_chunks_gives_no_match(client, scripted, ollama):
    scripted(("  ", {"source": "a.md"}, 0.2), ("far away", {"source": "b.md"}, 1.5))
    r = ask(client, n_results=2)
    assert r.json() == {"answer": api.NO_MATCH_ANSWER, "sources": [], "confidence": "low"}
    assert ollama.calls == []


# ── ChromaDB failures ──────────────────────────────────────────────────────

def test_chroma_unreadable_is_503(client, monkeypatch, ollama):
    monkeypatch.setattr(api, "collection", BrokenCollection())
    r = ask(client)
    assert r.status_code == 503
    assert r.json() == {"detail": "ChromaDB is not readable"}
    assert ollama.calls == []


def test_chroma_search_failure_is_503_and_logged(client, monkeypatch, ollama, caplog):
    # count works but the search doesn't, e.g. the embedding model can't load
    class SearchFails(BrokenCollection):
        def count(self):
            return 5

    monkeypatch.setattr(api, "collection", SearchFails("embedding model not found"))
    r = ask(client)
    assert r.status_code == 503
    assert r.json() == {"detail": "ChromaDB search failed"}
    assert "embedding model not found" in caplog.text
    assert ollama.calls == []
