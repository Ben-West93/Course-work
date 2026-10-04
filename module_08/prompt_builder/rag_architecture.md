# RAG Architecture

The five stages of a RAG pipeline, using the stack from Modules 7 and 8.

```
 ┌──────────────────────────────┐
 │ 1. USER INPUT                │  What: user types a natural-language question
 │    Streamlit text box / CLI  │  Tech: Streamlit, Python input
 └──────────────┬───────────────┘
                │  question: str
                ▼
 ┌──────────────────────────────┐
 │ 2. QUERY EMBEDDING           │  What: question is turned into a vector with the
 │    sentence-transformers     │        same model used to embed the documents
 │    all-MiniLM-L6-v2          │  Tech: sentence-transformers
 └──────────────┬───────────────┘
                │  query vector: list[float] (384 dims)
                ▼
 ┌──────────────────────────────┐
 │ 3. VECTOR DB RETRIEVAL       │  What: nearest-neighbour search over stored chunk
 │    ChromaDB collection       │        embeddings, returns top-k closest chunks
 │    (cosine distance)         │  Tech: ChromaDB (PersistentClient)
 └──────────────┬───────────────┘
                │  chunks: list[{"text", "source", "distance"}]
                ▼
 ┌──────────────────────────────┐
 │ 4. PROMPT ASSEMBLY           │  What: system prompt + chunks labelled
 │    prompt_builder.py         │        [Source: file] + user question,
 │                              │        token budget checked (chars // 4)
 │                              │  Tech: Python (build_prompt / format_chunks)
 └──────────────┬───────────────┘
                │  prompt: str (~token estimate)
                ▼
 ┌──────────────────────────────┐
 │ 5. LLM GENERATION            │  What: model answers from the context only,
 │    chat completion API       │        cites sources, or says it doesn't
 │                              │        have enough information
 │                              │  Tech: LLM API (served via Docker in Module 8)
 └──────────────┬───────────────┘
                │  answer: str + cited sources
                ▼
           shown to user
```

Offline step (runs before any query): documents in `docs/` are chunked, embedded
with the same all-MiniLM-L6-v2 model, and written into ChromaDB by `ingest.py`.
Stage 2 has to use the same model as ingestion, otherwise the query vector and the
stored vectors live in different spaces and retrieval returns junk.

## Data passed between stages

| From → To | Data | Shape |
|---|---|---|
| 1 → 2 | user question | `str` |
| 2 → 3 | query embedding | `list[float]`, 384 dims |
| 3 → 4 | top-k chunks | `list[dict]` with `text`, `source`, `distance` |
| 4 → 5 | assembled prompt | `str` |
| 5 → user | grounded answer | `str` with `(Source: file)` citations |
