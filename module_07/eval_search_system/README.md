# Evaluate Your Search System

Module 07 exercise: measure the quality of a ChromaDB semantic search with precision and recall across several retrieval settings.

## Files

| File | Purpose |
|------|---------|
| `my_eval.py` | Completed evaluation script |
| `test_eval.py` | Multi-angle tests for edge cases and average calculations |
| `starter.py` | Original starter scaffold, kept for reference |

## Running

From the repo root:

```bash
.venv/bin/python module_07/eval_search_system/my_eval.py
```

To run the tests:

```bash
.venv/bin/python module_07/eval_search_system/test_eval.py
```

The script first prints an edge-case pre-check that lists each failure mode it guards against, then runs the evaluations and the analysis. It uses an in-memory `chromadb.EphemeralClient()`, so no files are written. The default embedding model is `all-MiniLM-L6-v2`.

## Dataset

**Corpus:** 16 one-sentence documents (`doc_01` to `doc_16`), 4 per topic, each tagged with `{"topic": ...}` metadata.

| IDs | Topic |
|-----|-------|
| doc_01–04 | FastAPI (validation, params, docs, dependencies) |
| doc_05–08 | Streamlit (reruns, session state, caching, calling an API) |
| doc_09–12 | Vector DBs (ChromaDB query, cosine distance, chunking, metadata filters) |
| doc_13–16 | HTML/CSS (Flexbox, Grid, media queries, semantic HTML) |

The collection uses cosine distance (`hnsw:space = cosine`, range 0–2, lower = more similar) in place of Chroma's default L2 distance, so a threshold value means the same thing for every query.

**Eval set:** 8 queries, each with ground-truth `relevant_ids`. The queries mix easy keyword matches (Q1, Q2) with paraphrased (Q5), cross-topic (Q6) and vague (Q8) queries. At load time the script asserts that each `relevant_ids` list is non-empty and that every ID exists in the corpus.

## Methodology and metrics

For each query, the script retrieves the top `n_results`, optionally drops hits with `distance > distance_threshold` (a hit exactly at the threshold is kept), and then computes:

- **Precision** = |retrieved ∩ relevant| / |retrieved|, the fraction of returned docs that are relevant.
- **Recall** = |retrieved ∩ relevant| / |relevant|, the fraction of relevant docs that were returned.

If the threshold removes every hit, |retrieved| = 0, and both metrics are set to 0 instead of raising `ZeroDivisionError`. The averages are plain means of the per-query values.

## Results

| Setting | Avg precision | Avg recall |
|---------|---------------|------------|
| n_results=3 | 54.2% | 95.8% |
| n_results=5 | 35.0% | 100.0% |
| n_results=5, threshold=0.5 | 12.5% | 25.0% |
| n_results=5, threshold=0.7 | **67.5%** | 70.8% |

Key findings (the script prints the full analysis):

- **Ranking is strong.** In the unthresholded runs, every relevant doc but one (Flexbox for Q3, at rank 4) lands in the top 3. Low precision at fixed k is largely a ceiling effect: a query with one relevant doc can score at most 1/k.
- **0.5 is too strict for MiniLM.** Paraphrased queries score 0.5–0.8, so 6 of 8 queries return nothing at that threshold. 0.7 gives the best precision/recall balance.
- **Weakest queries:** Q8, which is vague (the correct doc ranks 1st but at 0.735, so any threshold ≤ 0.7 drops it), Q3, which needs a doc with no shared wording, and Q5/Q6, which are paraphrased or cross-topic and keep only half their relevant docs at 0.7.
- **Suggested improvements:** hybrid BM25 + vector search, a cutoff relative to the best score instead of one absolute threshold, a reranker or stronger embedding model, and topic-prefixed chunks with metadata filters.

## Edge cases tested (`test_eval.py`)

- Thresholds of 0.0 and 0.1, where no documents pass: averages are 0%/0% and no division error occurs.
- A threshold exactly equal to a hit's distance keeps that hit. A threshold 1e-6 below it drops the hit.
- Broad versus specific queries: the broad query "web development" scores P=100% with low recall. Exact-identifier queries such as `st.cache_data` find the right doc at rank 1, but fall outside the 0.5 threshold, which is a case for hybrid search.
- `n_results=50`, which is larger than the 16-document collection, is clamped and returns recall of 100%.
- A ground-truth ID that is not in the corpus (`doc_99`) is rejected by the load-time assertion.
- For all four settings, the printed averages match means recomputed by hand from the per-query lines.
