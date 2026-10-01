# Module 07 — Expanded Semantic Search App

A Streamlit + ChromaDB app that searches the course notes in `docs/` by meaning, with source filtering, result counts, and expandable full text.

## Setup

```bash
cd ~/Course-work
.venv/bin/python -m pip install streamlit chromadb sentence-transformers
.venv/bin/streamlit run module_07/semantic_search_app/semantic_search_app.py
```

The first launch downloads `all-MiniLM-L6-v2` (~80 MB) and indexes `docs/` automatically. The vector store is written to `module_07/semantic_search_app/chroma_db/` (git-ignored), so it works from any launch directory.

## Using the app

- **Search box**: type a question or phrase. Empty or whitespace-only input shows no results.
- **Sidebar → Search only in these files**: pick one or more files to filter by. Leave it empty to search everything.
- **Sidebar → Re-index Documents**: re-reads `docs/` after you add or edit files. It is safe to run repeatedly.
- **Result card**: shows the source file, a relevance badge, the distance, a 150-character preview, a "Show full text" expander, and a "Similar to this" button.

## Features

| Feature | How it works |
|---|---|
| Caching | `@st.cache_resource` loads the model and ChromaDB collection (`docs_search`) once per server process |
| Chunking | Sliding window of 500 chars with 100-char overlap. IDs look like `fastapi_basics.md_chunk_0`, with metadata `{"source": filename}` |
| Indexing | `upsert()` runs when the collection is empty or when you click Re-index. Stale chunk IDs are deleted, so re-indexing is idempotent |
| Source filter | `where={"source": {"$in": selected}}` is added only for a real subset. Empty or all-selected omits `where` |
| Result count | "Showing X of Y total documents" (Y = `collection.count()`) |
| Relevance badge | 🟢 High < 0.5 · 🟡 Medium < 1.0 · 🔴 Low ≥ 1.0 (L2 distance, 3 d.p.) |
| Sidebar stats | Unique source files indexed, plus total chunks |
| Similar to this (bonus) | An `on_click` callback writes the chunk into `st.session_state["query"]`, and Streamlit re-runs the search |

## Documents (`docs/`)

`fastapi_basics.md`, `streamlit_guide.md`, `vector_databases.md`, `embeddings_explained.md`, `html_css_fundamentals.md`, `sql_databases.txt`: 6 files, 33 chunks.

## Sample queries

| Query | Filter | Top result |
|---|---|---|
| How does Streamlit keep state between reruns? | none | `streamlit_guide.md` (🟡 0.858) |
| How does Streamlit keep state between reruns? | `sql_databases.txt` | only SQL chunks returned |
| CSS flexbox layout → *Similar to this* on result 1 | none | same chunk (🟢 0.000), then neighbouring HTML/CSS chunks |
| how do I protect against SQL injection | none | `sql_databases.txt` (🔴 1.334, the parameterized-query chunk) |
| zxqv purple elephant tax returns | none | still 5 results, all 🔴 Low (≥ 1.6). Vector search always returns the nearest neighbours |

With L2 distance on MiniLM embeddings, correct matches often score 1.0–1.4, so a 🔴 badge doesn't always mean the result is wrong.

Other queries to try: "how do I validate request bodies", "why split text into overlapping pieces".

`starter.py` is the original TODO scaffold, kept for reference.
