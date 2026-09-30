# Module 07 — Personal Knowledge Base

A persistent ChromaDB collection of course notes with metadata-filtered semantic search.

## Setup

```bash
cd ~/Course-work
.venv/bin/python -m pip install chromadb
.venv/bin/python module_07/knowledge_base/knowledge_base.py   # run from the repo root or this folder
```

The first run downloads the default `all-MiniLM-L6-v2` embedding model (~80 MB). The store is written to `./chroma_db` relative to where you run the script (git-ignored). `upsert()` makes re-runs safe: the count stays at 18.

## Collection schema

- Collection: `my_knowledge` (default embedding function, L2 distance — lower is better)
- IDs: `doc_01` … `doc_18`
- Metadata: `{"module": "<string>", "topic": "<string>"}` — module is a string so filters like `{"module": "5"}` match.

| Module | Topic | Docs |
|---|---|---|
| 3 | database | 4 |
| 5 | api | 5 |
| 6 | frontend | 4 |
| 7 | ai | 5 |

## Sample query results

| Query | Filter | Top result (distance) |
|---|---|---|
| "how do computers understand meaning of text" | none | doc_14, embeddings (1.0944) |
| "sending requests" | module 5 | doc_07, HTTP status codes (1.2041) |
| "how to handle bad user input" | none | doc_06, Pydantic validation (1.7299) |

The paraphrased query shares almost no words with "Pydantic validates request data…" yet still ranks it first.

## Edge cases handled

- `module_filter="99"` (no docs) → prints "(no results)".
- Empty / whitespace query → skipped with a message instead of erroring.
- Special characters / SQL-style strings → treated as plain text, return results.
- Integer filter (`6`) → cast with `str()`, matches.
- `n_results` larger than the matching docs → returns only what exists.
- Two back-to-back runs → no duplicate-ID errors, count unchanged.
