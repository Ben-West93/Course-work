"""
L7 — Similarity Threshold Experiment  (STARTER)
================================================
Run with:
    python starter.py

Your goal: show how different similarity thresholds change which results are
returned for the same query.

Required features:
    1. A `knowledge_base` list of at least 15 sentences across 3–4 distinct
       topics (e.g. Python, cooking, space, music)
    2. A `test_queries` list of at least 5 queries — some clearly on one topic,
       some ambiguous
    3. Embed all knowledge-base sentences once
    4. For each query, compute similarity scores against all documents
    5. Display results at three thresholds: 0.3, 0.5, and 0.7
       For each threshold show: count of results, which sentences pass,
       and a note about any relevant sentences that are missed

Key concepts:
    Filtering by threshold:
        passing = [(score, sentence)
                   for score, sentence in zip(scores, knowledge_base)
                   if score >= threshold]
        passing.sort(reverse=True)

    Expected output format:
        Query: "How do I deploy my Python app?"

          Threshold 0.3: 6 results
            [0.72] Deploy FastAPI with uvicorn and Docker
            ...
          Threshold 0.5: 3 results
            ...
          Threshold 0.7: 1 result
            ...
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── Knowledge base ───────────────────────────────────────────────────────────
# TODO: Create `knowledge_base` — at least 15 sentences across 3–4 topics.
#       Aim for roughly equal coverage of each topic so threshold differences
#       are visible.

knowledge_base = [
    # Python / development
    # TODO: add 4–5 sentences

    # Another topic (e.g. cooking)
    # TODO: add 4–5 sentences

    # Another topic (e.g. space)
    # TODO: add 4–5 sentences

    # Optional 4th topic
]

# ── Test queries ─────────────────────────────────────────────────────────────
# TODO: Create `test_queries` — at least 5 queries.
#       Include some that clearly match one topic and some that are ambiguous.

test_queries = [
    # TODO: add your queries here
]

# ── Thresholds ───────────────────────────────────────────────────────────────
THRESHOLDS = [0.3, 0.5, 0.7]

# ── Model & embeddings ───────────────────────────────────────────────────────
print("Loading model and embedding knowledge base...")
# TODO: Load SentenceTransformer('all-MiniLM-L6-v2')
# TODO: Encode knowledge_base into `doc_embeddings`

# ── Run experiment ───────────────────────────────────────────────────────────
for query in test_queries:
    print(f'\nQuery: "{query}"')

    # TODO: Encode the query
    # TODO: Compute cosine similarity scores (1-D array)

    for threshold in THRESHOLDS:
        # TODO: Filter (score, sentence) pairs where score >= threshold
        # TODO: Sort by score descending
        # TODO: Print the threshold, count, and each result
        #       e.g. "  Threshold 0.3: 4 results"
        #            "    [0.72] Deploy FastAPI with uvicorn and Docker"
        pass
