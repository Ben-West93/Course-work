# my_rag

A RAG pipeline over my course notes. It loads the files in `docs/`, stores them in ChromaDB, pulls the 3 closest chunks for each question and streams an answer from Ollama (`llama3.2:1b`).

`starter.py` is the original scaffold. `my_rag.py` is the finished version.

## Running

```
ollama serve
ollama pull llama3.2:1b
python my_rag.py
```

Type a question at the `You:` prompt. `quit`, `exit` or `q` ends the loop, and so does Ctrl+C/Ctrl+D.

## Pipeline

```
docs/*.md, *.txt
   -> load_documents()  split on blank lines, one chunk per paragraph
   -> ingest()          upsert into the "course_docs" collection
question
   -> retrieve()        top 3 chunks by L2 distance (all-MiniLM-L6-v2)
   -> build_messages()  system prompt + [Source: file] tagged context
   -> generate()        POST /api/chat, stream=True
```

## Ingestion

- Only `.txt` and `.md` files are read, in sorted order. Windows line endings are converted first.
- Each file is split on `"\n\n"`, then stripped, and empty pieces are dropped. Every chunk gets the id `<filename>_<i>` and the metadata `{"source", "chunk_index"}`.
- `upsert` is used rather than `add`, so re-ingesting overwrites existing chunks instead of throwing duplicate-id errors.
- If the docs folder is missing or empty, you get a warning and 0 chunks.

The corpus has 7 files and 37 chunks. The topics are embeddings, ChromaDB, chunking, RAG, Ollama, prompt injection and semantic search thresholds.

## ChromaDB persistence

`chromadb.PersistentClient` writes to `rag_db/` next to the script. The path is built from `__file__` rather than the working directory, so running from the repo root and running from this folder both use the same DB. Ingestion only happens when `collection.count() == 0`. Later runs reuse the stored embeddings. If you change the docs, delete `rag_db/` to rebuild it. The folder is gitignored.

`retrieve()` clamps `n_results` to `collection.count()` and returns `[]` when the collection is empty.

## Streaming

Ollama sends newline-delimited JSON. `generate()` reads it with `iter_lines()`, skips blank or unparseable lines, prints each `message.content` token with `flush=True` and adds it to the returned string. It stops when it sees `"done": true`.

Errors:
- `ConnectionError` (server not running): returns a message telling you to run `ollama serve`. The loop keeps going.
- `HTTPError` (usually a model that hasn't been pulled): returns a message telling you to run `ollama pull`.

## Test results

| Type | Question | Top source (distance) | Result |
|---|---|---|---|
| Answerable | How does ChromaDB persist data between runs, and what is the difference between add and upsert? | chromadb.md (1.003) | Cited chromadb.md and explained PersistentClient and upsert correctly. It got `add` wrong ("does nothing" when the doc says it fails) and added performance claims that aren't in the docs. |
| Related, not in docs | What chunk size should I use for legal contracts? | chunking.txt (1.060) | "I don't have enough information to answer that." |
| Out of scope | What is the capital of Australia? | prompt_injection.md (1.745) | "I don't have enough information to answer that." |

With Ollama stopped, the same flow printed the retrieved chunks and then the connection error, without crashing.

Notes:
- The out-of-scope query's best match was at 1.745, compared with about 1.0 for the real hits. A distance cutoff around 1.4 (see semantic_search.md) could skip the LLM call entirely for questions like that.
- The 1b model follows the refusal rule well but still adds details on answerable questions. A larger model or a lower temperature would probably help.
