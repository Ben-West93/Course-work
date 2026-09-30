"""
L7 — Course Content Search
===========================
Run with:
    python course_search.py

An interactive semantic search tool over course-summary sentences.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── Knowledge base ──────────────────────────────────────────────────────────
sentences = [
    "HTML uses tags to define the structure of a web page",
    "CSS controls the visual layout and styling of HTML elements",
    "JavaScript adds interactivity and dynamic behavior to web pages",
    "FastAPI automatically validates request data using Pydantic models",
    "Streamlit re-runs the entire script from top to bottom when any widget is interacted with",
    "Git tracks changes to files and enables collaboration through branches and commits",
    "SQL databases store data in tables made up of rows and columns",
    "REST APIs expose resources through HTTP methods like GET, POST, PUT, and DELETE",
    "Sentence embeddings convert text into numeric vectors that capture semantic meaning",
    "Cosine similarity measures how closely two vectors point in the same direction",
    "Python virtual environments isolate project dependencies from the global system",
    "Docker containers package an application with everything it needs to run consistently",
]

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model():
    """Lazily load and cache the SentenceTransformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_sentences(model, sentence_list):
    """Encode a list of sentences into embeddings."""
    return model.encode(sentence_list)


def search(query, model, doc_embeddings, corpus, top_n=3):
    """
    Return the top_n (score, sentence) pairs from `corpus` that best match `query`.

    Returns an empty list for a blank/whitespace-only query.
    """
    if query is None or not query.strip():
        return []

    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]

    top_indices = np.argsort(scores)[::-1][:top_n]
    return [(scores[i], corpus[i]) for i in top_indices]


def format_results(results):
    """Format (score, sentence) pairs into display lines."""
    return [f"  {i + 1}. [{score:.4f}] {sentence}" for i, (score, sentence) in enumerate(results)]


def main():
    print("Loading model...")
    model = get_model()

    print(f"Embedding {len(sentences)} sentences...")
    doc_embeddings = embed_sentences(model, sentences)

    print("\nSemantic Search — type 'quit' to exit\n")

    while True:
        query = input("Search: ").strip()

        if query.lower() == "quit":
            break

        if not query:
            continue

        results = search(query, model, doc_embeddings, sentences)

        print("\nTop 3 results:")
        for line in format_results(results):
            print(line)
        print()


if __name__ == "__main__":
    main()
