# Chunking Experiment Results

## Summary

Precision/recall are source-level over the top 5 results. Top-3 relevant counts how many of the first 3 chunks came from a relevant document.

| Chunk size | Overlap | Chunks | Avg precision | Avg recall | Top-1 accuracy | Top-3 relevant | Mean top-3 score |
|---|---|---|---|---|---|---|---|
| 150 | 30 | 170 | 0.80 | 1.00 | 100% | 14/15 | 0.473 |
| 300 | 60 | 87 | 0.63 | 1.00 | 100% | 15/15 | 0.408 |
| 600 | 120 | 44 | 0.77 | 1.00 | 100% | 14/15 | 0.397 |

## Per-query comparison

| Query | 150 chars: top-3 relevant / mean score | 300 chars: top-3 relevant / mean score | 600 chars: top-3 relevant / mean score |
|---|---|---|---|
| 1. How do I keep a variable's value between reruns in Streamlit? | 3/3 / 0.432 | 3/3 / 0.429 | 3/3 / 0.414 |
| 2. Which HTTP status code means the user is logged in but not allowed to access something? | 3/3 / 0.505 | 3/3 / 0.395 | 3/3 / 0.379 |
| 3. How can my API automatically reject requests with bad data? | 3/3 / 0.464 | 3/3 / 0.422 | 3/3 / 0.397 |
| 4. Why are dictionary lookups faster than searching a list? | 2/3 / 0.540 | 3/3 / 0.463 | 2/3 / 0.488 |
| 5. How do I make several database writes all succeed or all fail together? | 3/3 / 0.424 | 3/3 / 0.332 | 3/3 / 0.305 |

## Top-3 results per query

### 1. How do I keep a variable's value between reruns in Streamlit?

Relevant: streamlit.txt

| Chunk size | Rank | Source | Chunk | Score | Relevant |
|---|---|---|---|---|---|
| 150 | 1 | streamlit.txt | 2 | 0.470 | yes |
| 150 | 2 | streamlit.txt | 16 | 0.441 | yes |
| 150 | 3 | streamlit.txt | 7 | 0.384 | yes |
| 300 | 1 | streamlit.txt | 1 | 0.540 | yes |
| 300 | 2 | streamlit.txt | 8 | 0.417 | yes |
| 300 | 3 | streamlit.txt | 3 | 0.329 | yes |
| 600 | 1 | streamlit.txt | 1 | 0.490 | yes |
| 600 | 2 | streamlit.txt | 0 | 0.414 | yes |
| 600 | 3 | streamlit.txt | 4 | 0.339 | yes |

### 2. Which HTTP status code means the user is logged in but not allowed to access something?

Relevant: rest-apis.txt

| Chunk size | Rank | Source | Chunk | Score | Relevant |
|---|---|---|---|---|---|
| 150 | 1 | rest-apis.txt | 5 | 0.599 | yes |
| 150 | 2 | rest-apis.txt | 7 | 0.470 | yes |
| 150 | 3 | rest-apis.txt | 6 | 0.446 | yes |
| 300 | 1 | rest-apis.txt | 3 | 0.457 | yes |
| 300 | 2 | rest-apis.txt | 7 | 0.366 | yes |
| 300 | 3 | rest-apis.txt | 1 | 0.361 | yes |
| 600 | 1 | rest-apis.txt | 1 | 0.454 | yes |
| 600 | 2 | rest-apis.txt | 0 | 0.344 | yes |
| 600 | 3 | rest-apis.txt | 2 | 0.338 | yes |

### 3. How can my API automatically reject requests with bad data?

Relevant: fastapi.txt, rest-apis.txt

| Chunk size | Rank | Source | Chunk | Score | Relevant |
|---|---|---|---|---|---|
| 150 | 1 | rest-apis.txt | 17 | 0.513 | yes |
| 150 | 2 | rest-apis.txt | 6 | 0.449 | yes |
| 150 | 3 | rest-apis.txt | 16 | 0.431 | yes |
| 300 | 1 | rest-apis.txt | 3 | 0.434 | yes |
| 300 | 2 | fastapi.txt | 1 | 0.422 | yes |
| 300 | 3 | rest-apis.txt | 8 | 0.411 | yes |
| 600 | 1 | fastapi.txt | 0 | 0.439 | yes |
| 600 | 2 | rest-apis.txt | 3 | 0.378 | yes |
| 600 | 3 | rest-apis.txt | 4 | 0.375 | yes |

### 4. Why are dictionary lookups faster than searching a list?

Relevant: python-advanced.txt

| Chunk size | Rank | Source | Chunk | Score | Relevant |
|---|---|---|---|---|---|
| 150 | 1 | python-advanced.txt | 4 | 0.713 | yes |
| 150 | 2 | python-advanced.txt | 3 | 0.479 | yes |
| 150 | 3 | embeddings-and-vectors.txt | 10 | 0.428 | no |
| 300 | 1 | python-advanced.txt | 1 | 0.554 | yes |
| 300 | 2 | python-advanced.txt | 2 | 0.430 | yes |
| 300 | 3 | python-advanced.txt | 4 | 0.404 | yes |
| 600 | 1 | python-advanced.txt | 0 | 0.554 | yes |
| 600 | 2 | python-advanced.txt | 1 | 0.494 | yes |
| 600 | 3 | embeddings-and-vectors.txt | 1 | 0.414 | no |

### 5. How do I make several database writes all succeed or all fail together?

Relevant: sql-databases.txt

| Chunk size | Rank | Source | Chunk | Score | Relevant |
|---|---|---|---|---|---|
| 150 | 1 | sql-databases.txt | 12 | 0.501 | yes |
| 150 | 2 | sql-databases.txt | 14 | 0.404 | yes |
| 150 | 3 | sql-databases.txt | 13 | 0.365 | yes |
| 300 | 1 | sql-databases.txt | 6 | 0.405 | yes |
| 300 | 2 | sql-databases.txt | 7 | 0.334 | yes |
| 300 | 3 | sql-databases.txt | 10 | 0.257 | yes |
| 600 | 1 | sql-databases.txt | 3 | 0.382 | yes |
| 600 | 2 | sql-databases.txt | 0 | 0.267 | yes |
| 600 | 3 | sql-databases.txt | 1 | 0.267 | yes |

