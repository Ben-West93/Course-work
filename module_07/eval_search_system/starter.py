"""
L7 — Evaluate Your Search System  (STARTER)
=============================================
Run with:
    python starter.py

Your goal: build an evaluation framework that measures precision and recall
for a semantic search system across multiple settings.

Required features:
    1. A ChromaDB collection with at least 12 documents across 3–4 topics
    2. An evaluation set: at least 6 test queries, each with a list of
       expected relevant document IDs
    3. An evaluate() function that computes per-query and average
       precision & recall
    4. Results at 3 different settings (vary threshold, n_results, or both)
    5. A written analysis in the output

Key concepts:
    Precision — of the results returned, how many were relevant?
        precision = len(relevant ∩ retrieved) / len(retrieved)

    Recall — of all relevant documents, how many did we retrieve?
        recall = len(relevant ∩ retrieved) / len(relevant)

    ChromaDB returns distances (lower = more similar).
    Convert to similarity for thresholding:
        similarity = 1 - distance   (for cosine distance)
        OR just filter by distance <= threshold

    Example evaluation set entry:
        {
            "query": "How does FastAPI validate request data?",
            "relevant_ids": ["doc_03", "doc_07"]
        }
"""

import chromadb

# ── Documents ─────────────────────────────────────────────────────────────────
# TODO: Define ids, documents, and metadatas lists (at least 12 items)
#       Cover 3–4 topics so that some queries are easy and some are hard.
#       Use unique IDs like "doc_01", "doc_02", ... — these are what the
#       evaluation set references.

ids = [
    # TODO
]

documents = [
    # TODO
]

metadatas = [
    # TODO — at least {"topic": "..."} for each document
]

# ── ChromaDB setup ────────────────────────────────────────────────────────────
# TODO: Create a chromadb.EphemeralClient() (in-memory, no files needed)
# TODO: Create or get a collection called "eval_collection"
# TODO: Upsert all documents

# ── Evaluation set ────────────────────────────────────────────────────────────
# TODO: Define eval_set — a list of dicts, each with:
#   "query":        the natural language query string
#   "relevant_ids": list of document IDs that are truly relevant

eval_set = [
    # TODO — at least 6 entries
    # {"query": "...", "relevant_ids": ["doc_01", "doc_05"]},
]


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

    for i, entry in enumerate(eval_set, 1):
        query        = entry["query"]
        relevant_ids = set(entry["relevant_ids"])

        # TODO: Query the collection with query_texts=[query], n_results=n_results
        #       include ["ids", "distances"]
        results = None  # replace with actual query

        # TODO: Extract retrieved IDs from results["ids"][0]
        retrieved_ids = []

        # TODO: If distance_threshold is set, filter out IDs whose distance
        #       exceeds the threshold

        # TODO: Compute precision and recall
        #   overlap   = len(relevant_ids ∩ set(retrieved_ids))
        #   precision = overlap / len(retrieved_ids)  (handle zero division)
        #   recall    = overlap / len(relevant_ids)   (handle zero division)
        precision = 0.0
        recall    = 0.0

        precisions.append(precision)
        recalls.append(recall)

        print(f"  Query {i}: P={precision*100:.1f}%  R={recall*100:.1f}%  "
              f"| '{query[:50]}'")

    avg_p = sum(precisions) / len(precisions) if precisions else 0
    avg_r = sum(recalls)    / len(recalls)    if recalls    else 0
    print(f"  AVERAGE: P={avg_p*100:.1f}%  R={avg_r*100:.1f}%")
    return avg_p, avg_r


# ── Run at 3 settings ─────────────────────────────────────────────────────────
# TODO: Call evaluate() with 3 different combinations of n_results / threshold
#       e.g. evaluate(n_results=3)
#            evaluate(n_results=5)
#            evaluate(n_results=5, distance_threshold=0.5)


# ── Analysis ─────────────────────────────────────────────────────────────────
print("\n=== ANALYSIS ===")
# TODO: Print your observations as a multi-line string or separate print() calls.
#       Answer:
#         - Which queries consistently performed well? Why?
#         - Which queries failed? What caused the low score?
#         - What would you change to improve the weakest queries?
#         - How did increasing n_results affect precision vs recall?
print("TODO: write your analysis here")
