"""
Edge-case tests for course_search.py

Run with:
    pytest test_course_search.py -v
or:
    python test_course_search.py
"""

import numpy as np
import pytest

from course_search import search, sentences, get_model, embed_sentences


@pytest.fixture(scope="module")
def model():
    return get_model()


@pytest.fixture(scope="module")
def doc_embeddings(model):
    return embed_sentences(model, sentences)


def test_empty_string_returns_no_results(model, doc_embeddings):
    results = search("", model, doc_embeddings, sentences)
    assert results == []


def test_whitespace_only_string_returns_no_results(model, doc_embeddings):
    results = search("   ", model, doc_embeddings, sentences)
    assert results == []


def test_non_ascii_characters(model, doc_embeddings):
    results = search("¿Cómo se usa CSS para diseñar páginas web? 你好世界", model, doc_embeddings, sentences)
    assert len(results) == 3
    for score, sentence in results:
        assert isinstance(score, (float, np.floating))
        assert sentence in sentences


def test_single_word_query(model, doc_embeddings):
    results = search("Docker", model, doc_embeddings, sentences)
    assert len(results) == 3
    scores = [score for score, _ in results]
    assert scores == sorted(scores, reverse=True)


def test_long_text_input(model, doc_embeddings):
    long_query = (
        "I would like to understand everything about how web pages are built, "
        "starting from the structure defined by markup languages, through the "
        "styling applied to elements, all the way to how servers validate and "
        "respond to incoming HTTP requests using modern frameworks, and how "
        "databases persist the resulting data across sessions. " * 5
    )
    results = search(long_query, model, doc_embeddings, sentences)
    assert len(results) == 3
    for score, sentence in results:
        assert -1.0 <= score <= 1.0
        assert sentence in sentences


def test_results_are_sorted_descending(model, doc_embeddings):
    results = search("How does FastAPI validate data?", model, doc_embeddings, sentences)
    scores = [score for score, _ in results]
    assert scores == sorted(scores, reverse=True)


def test_top_n_respected(model, doc_embeddings):
    results = search("git branches", model, doc_embeddings, sentences, top_n=5)
    assert len(results) == 5


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
