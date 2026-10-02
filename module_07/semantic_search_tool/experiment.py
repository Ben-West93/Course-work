"""
Module 7 Project — Semantic Search Tool
========================================
experiment.py — chunking experiment

Run with:
    python experiment.py
    python experiment.py --chunk-sizes 150 300 600 --overlap-ratio 0.2

For each chunk size the docs are re-indexed, the same EVAL_SET queries are run,
and the top-3 results (source, chunk, score, relevant?) are recorded together
with source-level precision/recall. The comparison is printed and written to
experiment_results.md. The index is restored to the default chunk size at the end.
"""

import argparse
from pathlib import Path

from evaluate import EVAL_SET, evaluate
from ingest import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, ingest

RESULTS_PATH = Path(__file__).resolve().parent / "experiment_results.md"
TOP_K = 3


def run_experiment(
    chunk_sizes: list[int], overlap_ratio: float = 0.2, n_results: int = 5
) -> dict:
    """
    Re-index at each chunk size and evaluate the same queries against each index.

    Overlap scales with chunk size (overlap = chunk_size * overlap_ratio) so every
    configuration shares the same proportion of context between chunks.

    Returns:
        {chunk_size: {"overlap", "chunks", "avg_precision", "avg_recall",
                      "top1_accuracy", "per_query": [...]}}
    """
    runs = {}
    for size in chunk_sizes:
        overlap = int(size * overlap_ratio)
        summary = ingest(chunk_size=size, overlap=overlap, verbose=False)
        scores = evaluate(n_results=n_results, verbose=False)
        runs[size] = {
            "overlap": overlap,
            "chunks": summary["chunks"],
            "avg_precision": scores["avg_precision"],
            "avg_recall": scores["avg_recall"],
            "top1_accuracy": scores["top1_accuracy"],
            "per_query": [
                {
                    "query": q["query"],
                    "precision": q["precision"],
                    "recall": q["recall"],
                    "top": [
                        {
                            "source": r["source"],
                            "chunk_index": r["chunk_index"],
                            "score": r["score"],
                            "relevant": r["source"] in item["relevant_sources"],
                        }
                        for r in q["results"][:TOP_K]
                    ],
                }
                for q, item in zip(scores["per_query"], EVAL_SET)
            ],
        }
    return runs


def mean_top_score(per_query_entry: dict) -> float:
    """Average similarity score of a query's top-K results (0.0 if none)."""
    top = per_query_entry["top"]
    return sum(r["score"] for r in top) / len(top) if top else 0.0


def relevant_in_top(per_query_entry: dict) -> int:
    """Number of the top-K results that come from a relevant source."""
    return sum(r["relevant"] for r in per_query_entry["top"])


def to_markdown(runs: dict, n_results: int) -> str:
    """Render the experiment runs as a Markdown report."""
    sizes = list(runs)
    lines = ["# Chunking Experiment Results", ""]

    lines += [
        "## Summary",
        "",
        f"Precision/recall are source-level over the top {n_results} results. "
        f"Top-{TOP_K} relevant counts how many of the first {TOP_K} chunks came "
        "from a relevant document.",
        "",
        "| Chunk size | Overlap | Chunks | Avg precision | Avg recall | Top-1 accuracy "
        f"| Top-{TOP_K} relevant | Mean top-{TOP_K} score |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for size in sizes:
        run = runs[size]
        relevant = sum(relevant_in_top(q) for q in run["per_query"])
        possible = sum(len(q["top"]) for q in run["per_query"])
        mean_score = sum(mean_top_score(q) for q in run["per_query"]) / len(run["per_query"])
        lines.append(
            f"| {size} | {run['overlap']} | {run['chunks']} | {run['avg_precision']:.2f} "
            f"| {run['avg_recall']:.2f} | {run['top1_accuracy']:.0%} "
            f"| {relevant}/{possible} | {mean_score:.3f} |"
        )

    lines += ["", "## Per-query comparison", ""]
    header = "| Query | " + " | ".join(
        f"{s} chars: top-{TOP_K} relevant / mean score" for s in sizes
    ) + " |"
    lines += [header, "|---" * (len(sizes) + 1) + "|"]
    for i, item in enumerate(EVAL_SET):
        cells = []
        for size in sizes:
            q = runs[size]["per_query"][i]
            cells.append(f"{relevant_in_top(q)}/{len(q['top'])} / {mean_top_score(q):.3f}")
        lines.append(f"| {i + 1}. {item['query']} | " + " | ".join(cells) + " |")

    lines += ["", f"## Top-{TOP_K} results per query", ""]
    for i, item in enumerate(EVAL_SET):
        lines += [f"### {i + 1}. {item['query']}", "",
                  f"Relevant: {', '.join(item['relevant_sources'])}", "",
                  "| Chunk size | Rank | Source | Chunk | Score | Relevant |",
                  "|---|---|---|---|---|---|"]
        for size in sizes:
            for rank, r in enumerate(runs[size]["per_query"][i]["top"], 1):
                lines.append(
                    f"| {size} | {rank} | {r['source']} | {r['chunk_index']} "
                    f"| {r['score']:.3f} | {'yes' if r['relevant'] else 'no'} |"
                )
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare chunk sizes")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[150, 300, 600])
    parser.add_argument("--overlap-ratio", type=float, default=0.2)
    parser.add_argument("--n-results", type=int, default=5)
    args = parser.parse_args()
    # Validate everything up front so a bad value cannot stop the run halfway.
    if any(size < 1 for size in args.chunk_sizes):
        parser.error("--chunk-sizes must all be at least 1")
    if not 0 <= args.overlap_ratio < 1:
        parser.error("--overlap-ratio must be at least 0 and less than 1")
    if args.n_results < 1:
        parser.error("--n-results must be at least 1")

    runs = run_experiment(args.chunk_sizes, args.overlap_ratio, args.n_results)
    report = to_markdown(runs, args.n_results)
    RESULTS_PATH.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nSaved to {RESULTS_PATH.name}")

    # Leave the app's index at the default configuration.
    ingest(chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP)
