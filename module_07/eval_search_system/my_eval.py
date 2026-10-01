"""
L7 — Evaluate Your Search System
=================================
Run with:
    python my_eval.py

Builds a 16-document ChromaDB collection across 4 topics, defines an
8-query evaluation set with ground-truth relevant IDs, and measures
precision and recall at several n_results / distance-threshold settings.
"""

import chromadb

# ── Documents ─────────────────────────────────────────────────────────────────
# 4 topics x 4 docs. Some docs deliberately overlap topics (e.g. doc_08 is a
# Streamlit doc that talks about calling an API) so a few queries are hard.

ids = [f"doc_{i:02d}" for i in range(1, 17)]

documents = [
    # FastAPI
    "FastAPI uses Pydantic models to validate request bodies; invalid JSON "
    "returns a 422 error with details about which fields failed.",
    "Path and query parameters in FastAPI are declared as function arguments "
    "with type hints, and FastAPI converts and validates them automatically.",
    "FastAPI generates interactive OpenAPI documentation at /docs using "
    "Swagger UI, built from your route definitions and models.",
    "Dependency injection in FastAPI uses Depends() to share logic such as "
    "database sessions or authentication checks across endpoints.",
    # Streamlit
    "Streamlit reruns the whole script from top to bottom every time a user "
    "interacts with a widget like a slider or button.",
    "st.session_state lets a Streamlit app remember values such as a counter "
    "or chat history between reruns.",
    "The @st.cache_data decorator caches the results of expensive functions "
    "like loading a CSV so Streamlit does not recompute them on each rerun.",
    "A Streamlit front end can call a backend REST API with the requests "
    "library and display the JSON response in a table.",
    # Vector databases
    "ChromaDB stores embeddings alongside documents and metadata, and "
    "collection.query returns the nearest neighbours with their distances.",
    "Cosine distance measures the angle between two embedding vectors; a "
    "smaller distance means the texts are more semantically similar.",
    "Chunking splits long documents into smaller passages before embedding "
    "so that each vector captures one focused idea.",
    "Metadata filters with a where clause narrow a vector search to documents "
    "matching fields such as topic or source.",
    # HTML / CSS
    "Flexbox arranges items in a single row or column, with justify-content "
    "and align-items controlling spacing and alignment.",
    "CSS Grid defines two-dimensional layouts with rows and columns using "
    "grid-template-columns and grid-template-rows.",
    "Media queries apply different CSS rules at different screen widths, "
    "which is the basis of responsive design for mobile devices.",
    "Semantic HTML elements like <header>, <nav>, <main> and <article> "
    "describe page structure and improve accessibility for screen readers.",
]

metadatas = (
    [{"topic": "fastapi"}] * 4
    + [{"topic": "streamlit"}] * 4
    + [{"topic": "vector_db"}] * 4
    + [{"topic": "html_css"}] * 4
)

# ── ChromaDB setup ────────────────────────────────────────────────────────────
# In-memory client, so every run starts from a clean collection.
# Cosine space is set explicitly: Chroma's default is L2, whose distances are
# not bounded the same way, so a threshold like 0.5 would mean something else.
client = chromadb.EphemeralClient()
collection = client.get_or_create_collection(
    name="eval_collection",
    metadata={"hnsw:space": "cosine"},
)
collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

# ── Evaluation set ────────────────────────────────────────────────────────────
# Mix of easy (keyword overlap with one topic), multi-doc, and hard queries
# (paraphrased, cross-topic, or vague).

eval_set = [
    {"query": "How does FastAPI validate request data?",
     "relevant_ids": ["doc_01", "doc_02"]},
    {"query": "How do I keep a value between reruns in Streamlit?",
     "relevant_ids": ["doc_06"]},
    {"query": "How do I build a responsive page layout with CSS?",
     "relevant_ids": ["doc_13", "doc_14", "doc_15"]},
    {"query": "What does a distance score mean in a vector search?",
     "relevant_ids": ["doc_09", "doc_10"]},
    {"query": "My app is slow because it reloads data every time",
     "relevant_ids": ["doc_05", "doc_07"]},
    {"query": "Connect a dashboard front end to a Python backend service",
     "relevant_ids": ["doc_08", "doc_03"]},
    {"query": "Should I split big files into pieces before indexing?",
     "relevant_ids": ["doc_11"]},
    {"query": "How can I restrict results to a specific category?",
     "relevant_ids": ["doc_12"]},
]

# ── Sanity checks on the eval set ─────────────────────────────────────────────
# Catch unindexed or empty ground truth up front: a typo'd ID can never be
# retrieved and would silently cap recall below 100%.
assert len(ids) == len(set(ids)), "duplicate document IDs"
_known = set(ids)
for _entry in eval_set:
    assert _entry["relevant_ids"], f"empty relevant_ids: {_entry['query']}"
    _missing = set(_entry["relevant_ids"]) - _known
    assert not _missing, f"unindexed IDs {_missing} in: {_entry['query']}"


# ── Evaluation function ────────────────────────────────────────────────────────
def evaluate(n_results: int, distance_threshold: float = None):
    """
    Run every query in eval_set, compute precision and recall for each,
    and print per-query results plus averages.

    Args:
        n_results:          how many results to retrieve per query
        distance_threshold: if set, only count results with distance <= this
    """
    print(f"\n=== Evaluation: n_results={n_results}, threshold={distance_threshold} ===")

    precisions = []
    recalls    = []

    # Chroma can't return more items than the collection holds.
    k = min(n_results, collection.count())

    for i, entry in enumerate(eval_set, 1):
        query        = entry["query"]
        relevant_ids = set(entry["relevant_ids"])

        results = collection.query(
            query_texts=[query], n_results=k, include=["distances"]
        )  # ids are always returned

        retrieved_ids = results["ids"][0]
        distances     = results["distances"][0]

        # Threshold filter: cosine distance is 0 for identical direction and
        # grows as texts diverge, so keep only hits at or under the cutoff
        # (<= means a hit exactly on the boundary still counts).
        if distance_threshold is not None:
            retrieved_ids = [
                doc_id for doc_id, dist in zip(retrieved_ids, distances)
                if dist <= distance_threshold
            ]

        # Precision = |retrieved ∩ relevant| / |retrieved|  — how clean the results are.
        # Recall    = |retrieved ∩ relevant| / |relevant|   — how complete they are.
        overlap = len(relevant_ids & set(retrieved_ids))

        # Zero-division guards: a strict threshold can filter out every hit,
        # leaving nothing retrieved. Returning nothing earns no credit, so
        # both metrics are 0 rather than undefined.
        precision = overlap / len(retrieved_ids) if retrieved_ids else 0.0
        recall    = overlap / len(relevant_ids)  if relevant_ids  else 0.0

        precisions.append(precision)
        recalls.append(recall)

        print(f"  Query {i}: P={precision*100:.1f}%  R={recall*100:.1f}%  "
              f"| '{query[:50]}'")

    avg_p = sum(precisions) / len(precisions) if precisions else 0
    avg_r = sum(recalls)    / len(recalls)    if recalls    else 0
    print(f"  AVERAGE: P={avg_p*100:.1f}%  R={avg_r*100:.1f}%")

    return avg_p, avg_r


# ── Run at 3 settings ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    evaluate(n_results=3)
    evaluate(n_results=5)
    evaluate(n_results=5, distance_threshold=0.5)
    evaluate(n_results=5, distance_threshold=0.7)

    # ── Analysis ─────────────────────────────────────────────────────────────
    print("\n=== ANALYSIS ===")
    print("""
1. What worked consistently
   Q1 (FastAPI validation), Q2 (Streamlit reruns), Q4 (distance scores) and
   Q7 (splitting files) put a relevant doc at rank 1 every time and hit
   100% recall at n=3, n=5 and threshold 0.7. They share distinctive
   words with their target docs ("FastAPI", "validate", "reruns",
   "distance", "split"). Q1 is the strongest match in the set (top distance
   0.24) and the only multi-doc query to keep full recall at threshold 0.5.
   Their low precision at n=3/n=5 is mostly a metric ceiling, not a retrieval
   failure: a query with 1 relevant doc can score at most 1/k (33.3% at k=3,
   20% at k=5) no matter how good the ranking is.

2. What failed or was weak
   - Q3 (responsive layout): doc_13 (Flexbox) ranks 4th, behind the semantic
     HTML doc, because it never mentions "responsive" or "page layout".
     Recall is 66.7% at n=3 and at threshold 0.7.
   - Q8 (restrict to a category): the vaguest query, with no topic words.
     doc_12 still ranks 1st, but at distance 0.735, so any threshold <= 0.7
     drops it and the query scores 0/0 (nothing retrieved).
   - Q5 (app slow, reloads data) and Q6 (dashboard to backend) are
     paraphrased or cross-topic. One relevant doc in each pair scores
     0.6-0.7, but the other sits at 0.79-0.86 (for Q5 the session_state
     doc even outranks the rerun doc), so a 0.7 cutoff keeps only one of
     the two (recall 50%).

3. Effect of n_results and threshold
   n=3 -> n=5 raised average recall 95.8% -> 100% but cut precision
   54.2% -> 35.0%: the extra 2 slots only add noise once the relevant docs
   are already in. Threshold 0.5 was far too strict for all-MiniLM-L6-v2,
   whose paraphrase distances mostly fall in 0.5-0.8: 6 of 8 queries
   returned nothing, and averages collapsed to P=12.5% R=25.0%. Loosening to
   0.7 gave the best balance (P=67.5% R=70.8%) by dropping off-topic tails
   while keeping close matches.

4. How to improve the weak queries
   - Hybrid search (BM25 + vectors) so exact keywords like "Streamlit" or
     "cache" can rescue paraphrased queries like Q5.
   - A relative cutoff (keep hits within ~0.15 of the best distance) in
     place of one absolute threshold, since good scores vary per query (0.24
     for Q1 vs 0.735 for Q8).
   - A cross-encoder reranker over the top 10, or a stronger embedding model
     (e.g. bge or e5), to push doc_13 above doc_16 for Q3.
   - Enrich chunks with context (e.g. prefix the topic: "CSS layout:
     Flexbox ...") and use topic metadata filters when the user's topic is
     known; query expansion would help vague queries like Q8.
""")
