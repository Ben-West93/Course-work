import pytest

import my_rag_api as api
from my_rag_api import compute_confidence, unsupported_specifics


def chunks(*distances, text="ChromaDB stores 384-dimensional vectors for the API.", source="chromadb.md"):
    return [{"text": text, "source": source, "distance": d} for d in distances]


# ── distance thresholds ────────────────────────────────────────────────────

def test_no_chunks_is_low():
    assert compute_confidence([]) == "low"
    assert compute_confidence([], "Some answer", "Some question") == "low"


@pytest.mark.parametrize("distance, expected", [
    (0.0, "high"), (0.2, "high"), (0.4999, "high"),
    (0.5, "medium"), (0.65, "medium"), (0.7999, "medium"),
    (0.8, "low"), (1.0, "low"), (1.2, "low"), (3.9, "low"),
])
def test_thresholds(distance, expected):
    assert compute_confidence(chunks(distance)) == expected


def test_best_chunk_decides_regardless_of_order():
    assert compute_confidence(chunks(1.1, 0.3, 0.9)) == "high"
    assert compute_confidence(chunks(0.9, 1.1, 0.6)) == "medium"


def test_weak_extra_chunks_dont_drag_a_strong_match_down():
    # borderline chunks still go to the model as context, but a strong best
    # match is what the answer mostly comes from
    assert compute_confidence(chunks(0.3, 1.15, 1.19)) == "high"


# ── model says it can't answer ─────────────────────────────────────────────

@pytest.mark.parametrize("answer", [
    "I don't have enough information to answer that.",
    "I do not have enough information about the current version.",
    "I don’t have enough information.",
    "There is not enough information in the context.",
    "The context doesn't contain enough information about licensing.",
])
def test_refusal_is_low_even_with_a_close_match(answer):
    assert compute_confidence(chunks(0.1), answer, "What is ChromaDB?") == "low"


@pytest.mark.parametrize("answer", [
    "ChromaDB stores enough information in metadata to filter results.",
    "It doesn't matter which client you use. Enough information is kept either way.",
])
def test_enough_information_without_a_refusal_is_not_low(answer):
    assert compute_confidence(chunks(0.1), answer, "q") == "high"


# ── specifics that aren't in the sources ───────────────────────────────────

@pytest.mark.parametrize("distance, expected", [(0.3, "medium"), (0.6, "low"), (0.9, "low")])
def test_made_up_licence_drops_one_level(distance, expected):
    answer = "ChromaDB is released under the MIT License."
    assert compute_confidence(chunks(distance), answer, "What license is ChromaDB under?") == expected


def test_made_up_number_drops_one_level():
    answer = "MiniLM embeddings are 128-dimensional."
    assert compute_confidence(chunks(0.3), answer, "What dimension are the embeddings?") == "medium"


@pytest.mark.parametrize("answer, question", [
    ("They are 384-dimensional vectors.", "What size are the vectors?"),
    ("Embeddings are created by the API.", "How are embeddings made?"),
    ("Yes, RAG helps with that.", "Does RAG reduce hallucination?"),
    ("It has 3 stages.", "How many stages are there?"),
    ("Pass n_results=5 to get more.", "How do I get more results?"),
    ("Two LLMs were compared.", "Compare the LLM options"),
    ("See chromadb.md for details.", "Where is this documented?"),
])
def test_specifics_from_the_sources_or_question_are_fine(answer, question):
    assert unsupported_specifics(answer, question, chunks(0.3)) == set()
    assert compute_confidence(chunks(0.3), answer, question) == "high"


def test_numbers_with_commas_match_either_way():
    c = chunks(0.3, text="The context window is 32768 tokens, about 1,024 lines.")
    assert unsupported_specifics("It holds 32,768 tokens and 1024 lines.", "q", c) == set()


def test_acronym_inside_a_longer_word_doesnt_count():
    c = chunks(0.3, text="There is no limit, just submit and commit the data.")
    assert unsupported_specifics("It uses the MIT license.", "q", c) == {"MIT"}


def test_unsupported_specifics_lists_what_it_found():
    found = unsupported_specifics("Released under MIT in 2019 with 1,024 parameters.", "q", chunks(0.3))
    assert found == {"MIT", "2019", "1024"}


# ── through the endpoint ───────────────────────────────────────────────────

def test_endpoint_lowers_confidence_for_made_up_specifics(client, ingested, ollama):
    ollama.answer = "ChromaDB is released under the MIT License [Source: chromadb.md]."
    body = client.post("/ask", json={"question": "Is ChromaDB an open-source vector database?"}).json()
    retrieval_only = compute_confidence(body["sources"])
    assert body["sources"]
    assert api.LEVELS.index(body["confidence"]) == max(api.LEVELS.index(retrieval_only) - 1, 0)


def test_endpoint_refusal_is_low(client, ingested, ollama):
    ollama.answer = "I don't have enough information to answer that from the context."
    body = client.post("/ask", json={"question": "How does ChromaDB store embeddings?"}).json()
    assert body["sources"]
    assert body["confidence"] == "low"


# ── heading-only chunks (regression) ───────────────────────────────────────

def test_no_chunk_is_just_a_heading():
    for c in api.load_documents(api.DOCS_DIR):
        assert any(not line.startswith("#") for line in c["text"].splitlines() if line.strip()), c["id"]


@pytest.mark.parametrize("question", ["ChromaDB", "Embeddings", "Prompt injection"])
def test_one_word_question_gets_real_content_not_a_heading(client, ingested, question):
    top = client.post("/ask", json={"question": question}).json()["sources"][0]
    # used to come back as just "# ChromaDB" at distance 0.32 -> "high"
    assert len(top["text"]) > 60
    assert not api.is_heading(top["text"])
