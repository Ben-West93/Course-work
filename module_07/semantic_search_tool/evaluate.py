"""
Module 7 Project — Semantic Search Tool
========================================
evaluate.py — precision and recall for your search system

Run with:
    python evaluate.py
    python evaluate.py --n-results 5 --threshold 0.4

Define your evaluation set in EVAL_SET, then run this script against
different index configurations (different chunk sizes) to compare results.
"""

import argparse
from search import search, get_collection_stats


# Define your evaluation set here.
# Each entry needs a query and the source filenames you expect to be relevant.
# Use at least 5 queries for the chunking experiment.
EVAL_SET = [
    {
        "query": "How do I keep a variable's value between reruns in Streamlit?",
        "relevant_sources": ["streamlit.txt"],
    },
    {
        "query": "Which HTTP status code means the user is logged in but not allowed to access something?",
        "relevant_sources": ["rest-apis.txt"],
    },
    {
        "query": "How can my API automatically reject requests with bad data?",
        "relevant_sources": ["fastapi.txt", "rest-apis.txt"],
    },
    {
        "query": "Why are dictionary lookups faster than searching a list?",
        "relevant_sources": ["python-advanced.txt"],
    },
    {
        "query": "How do I make several database writes all succeed or all fail together?",
        "relevant_sources": ["sql-databases.txt"],
    },
]


def precision_recall(
    retrieved_sources: list[str], relevant_sources: list[str]
) -> tuple[float, float]:
    """
    Compute source-level precision and recall.

    Duplicate sources (several chunks from the same file) count once, so the
    metrics measure whether the *right documents* came back.

        precision = |retrieved ∩ relevant| / |retrieved|
        recall    = |retrieved ∩ relevant| / |relevant|

    Returns (precision, recall) as floats in [0, 1]. Empty inputs give 0.0.
    """
    retrieved = set(retrieved_sources)
    relevant = set(relevant_sources)
    hits = len(retrieved & relevant)
    precision = hits / len(retrieved) if retrieved else 0.0
    recall = hits / len(relevant) if relevant else 0.0
    return precision, recall


def evaluate(
    n_results: int = 5,
    distance_threshold: float = None,
    eval_set: list[dict] = None,
    verbose: bool = True,
) -> dict:
    """
    Run every query in EVAL_SET and print per-query and average precision/recall.

    Returns:
        {
            "per_query":     list of {"query", "precision", "recall", "top1_hit", "results"},
            "avg_precision": float,
            "avg_recall":    float,
            "top1_accuracy": float,  # share of queries whose #1 result is relevant
        }
    """
    eval_set = EVAL_SET if eval_set is None else eval_set
    if not eval_set:
        raise ValueError("EVAL_SET is empty. Add queries before evaluating.")

    per_query = []
    for item in eval_set:
        results = search(
            item["query"], n_results=n_results, distance_threshold=distance_threshold
        )
        precision, recall = precision_recall(
            [r["source"] for r in results], item["relevant_sources"]
        )
        top1_hit = bool(results) and results[0]["source"] in item["relevant_sources"]
        per_query.append({
            "query": item["query"],
            "precision": precision,
            "recall": recall,
            "top1_hit": top1_hit,
            "results": results,
        })

    count = len(per_query)
    summary = {
        "per_query": per_query,
        "avg_precision": sum(q["precision"] for q in per_query) / count,
        "avg_recall": sum(q["recall"] for q in per_query) / count,
        "top1_accuracy": sum(q["top1_hit"] for q in per_query) / count,
    }

    if verbose:
        stats = get_collection_stats()
        print(
            f"Index: {stats['total_chunks']} chunks, chunk_size={stats['chunk_size']}, "
            f"overlap={stats['overlap']} | n_results={n_results}, "
            f"threshold={distance_threshold}\n"
        )
        print(f"{'#':<3}{'Precision':>10}{'Recall':>8}{'Top-1':>7}  Query")
        for i, q in enumerate(per_query, 1):
            print(
                f"{i:<3}{q['precision']:>10.2f}{q['recall']:>8.2f}"
                f"{'yes' if q['top1_hit'] else 'no':>7}  {q['query']}"
            )
        print(
            f"\nAverage precision: {summary['avg_precision']:.2f} | "
            f"Average recall: {summary['avg_recall']:.2f} | "
            f"Top-1 accuracy: {summary['top1_accuracy']:.0%}"
        )

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate search quality")
    parser.add_argument("--n-results", type=int,   default=5)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    if args.n_results < 1:
        parser.error("--n-results must be at least 1")
    try:
        evaluate(n_results=args.n_results, distance_threshold=args.threshold)
    except ValueError as error:
        parser.error(str(error))
