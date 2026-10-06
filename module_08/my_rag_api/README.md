# my_rag_api

The RAG pipeline from `rag_guardrails` wrapped in a FastAPI app. It uses ChromaDB for retrieval and `llama3.2:1b` through Ollama for answers.

`starter.py` is the original scaffold. `my_rag_api.py` is the finished version.

## Running

```
ollama serve
uvicorn module_08.my_rag_api.my_rag_api:app --reload --port 8000   # from the repo root
uvicorn my_rag_api:app --reload --port 8000                        # or from this folder
```

`DB_PATH` and `DOCS_DIR` are resolved relative to the script, so both commands use the same `rag_db/` (gitignored) and `docs/` (7 files, 37 chunks).

CORS is open (`allow_origins=["*"]`, all methods and headers) so a separate frontend can call the API.

## Routes

| Method | Path | Returns |
|---|---|---|
| POST | `/ask` | `AskResponse`: an answer grounded in the retrieved chunks, plus the sources and a confidence level |
| POST | `/ingest` | `IngestResponse`: loads `docs/` into the `documents` collection |
| GET | `/stats` | `{"document_count", "model", "db_path"}` |
| GET | `/health` | `{"status", "chromadb", "ollama", "document_count", "model"}` |

## Schemas

```
AskRequest      question: str, n_results: int = 3, max_distance: float = 1.2
SourceChunk     text: str, source: str, distance: float
AskResponse     answer: str, sources: list[SourceChunk], confidence: str
IngestResponse  chunks_ingested: int, message: str
```

`question` has a `field_validator` that strips whitespace and raises `ValueError` if nothing is left. Pydantic turns that into a 422 before the route runs.

Chunks with `distance > max_distance` are dropped before the prompt is built. Confidence comes from the best kept distance: `high` below 0.5, `medium` below 0.8, otherwise `low`.

`n_results` is clamped to between 1 and the collection size, because Chroma raises an error outside that range.

## Status codes

| Case | Code | Body |
|---|---|---|
| Normal request | 200 | the response model |
| Empty or whitespace `question`, missing field, wrong type | 422 | Pydantic error list |
| Ollama not running (`ConnectionError`) | 503 | `{"detail": "Ollama service is unavailable"}` |
| Ollama timeout, or non-200 such as a missing model | 503 | `{"detail": "..."}` |
| Nothing ingested yet, or no chunk within `max_distance` | 200 | `"No relevant documents found..."`, `sources: []`, `confidence: "low"`. Ollama is not called |
| `/ingest` with a missing or empty `docs/` | 200 | `chunks_ingested: 0` and a message naming the folder |

Re-running `/ingest` upserts with stable ids (`filename_index`), so the count stays at 37 instead of doubling.

## Health logic

- `chromadb`: `"ok"` if `collection.count()` succeeds, otherwise `"error"`
- `ollama`: `"connected"` if `GET /api/tags` returns 200, `"disconnected"` on a connection error or timeout

| chromadb | ollama | status |
|---|---|---|
| ok | connected | `ok` |
| ok | disconnected | `degraded` (ingest and retrieval still work, `/ask` gives 503) |
| error | either | `error` |

`/health` always returns 200 so a monitor can read the body. It does not return a 503 itself.

## Testing in Swagger UI

1. Open http://localhost:8000/docs
2. `GET /stats`, then Try it out and Execute. `document_count` is 0 on a fresh db
3. `POST /ingest`, then Execute. You should get 37 chunks from 7 files. Run `/stats` again to see 37
4. `POST /ask` with `{"question": "What embedding model does ChromaDB use by default?"}`. You should get 200, the all-MiniLM-L6-v2 answer and `medium` confidence
5. `POST /ask` with `{"question": "   "}`. You should get a 422
6. Stop Ollama, then call `GET /health` (`"degraded"`, `"disconnected"`) and `POST /ask` again (503)

## Results

| Test | Result |
|---|---|
| `/stats` before ingest | `document_count: 0` |
| `/ingest`, then `/stats` | 37 chunks from 7 files, `document_count: 37` (still 37 after a second ingest) |
| `/ask` valid question | 200, "ChromaDB uses the all-MiniLM-L6-v2 sentence-transformer model by default.", sources at 0.541 / 0.683 / 0.891, `medium` |
| `/ask` `""` and `"   "` | 422, `question must not be empty` |
| `/health`, Ollama off | `degraded`, `disconnected`, 37 docs |
| `/ask`, Ollama off | 503 `Ollama service is unavailable` |
| `/health`, Ollama on | `ok`, `connected` |
| `/ask` "Who won the 2018 World Cup?" | no chunk within 1.2, so the no-documents answer with `low` |

The default `max_distance` of 1.2 is a little stricter than the 1.3 I settled on in `rag_guardrails`. Questions whose best match sits between 1.2 and 1.3 (for example the add vs upsert one at 1.214) now get the no-documents answer unless you raise `max_distance` in the request.
