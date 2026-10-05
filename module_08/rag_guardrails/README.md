# rag_guardrails

This is the `my_rag.py` pipeline with 4 reliability guardrails added. It uses the same `docs/` corpus (7 files, 37 chunks), ChromaDB and `llama3.2:1b`.

`starter.py` is the original scaffold. `rag_pipeline.py` is the finished version.

## Running

```
ollama serve
python rag_pipeline.py
```

On the first run it ingests `docs/` into `rag_db/` (gitignored). It then runs the 4 test queries and prints the structured result for each one.

## Changes from my_rag.py

The loading, ingest and retrieval code is the same. These are the changes:

- **Generation is non-streaming** (`stream: False`, `temperature: 0`), because the answer is returned inside a dict rather than printed token by token. Connection and HTTP failures raise `OllamaError`, which `rag_query` turns into an error answer instead of a crash.
- **The interactive loop is gone.** `__main__` runs the 4 test queries, as the starter does.

## Guardrails

**1. Distance threshold.** `filter_chunks()` drops any chunk with distance >= `DISTANCE_THRESHOLD`. If nothing is left, `rag_query` returns a fixed "couldn't find relevant information" answer and never calls the LLM.

The starter suggests 1.0, but that's too strict for this corpus. Measured top-1 distances:

| Query | Top distance | Relevant? |
|---|---|---|
| What embedding model does ChromaDB use by default? | 0.541 | yes |
| How does PersistentClient store data? | 0.960 | yes |
| What is the difference between add and upsert in ChromaDB? | 1.214 | yes |
| How many dimensions does all-MiniLM-L6-v2 produce? | 1.280 | yes |
| How do I make it more accurate? | 1.552 | no (vague) |
| What is the capital of Australia? | 1.745 | no |
| Who won the 2018 World Cup? | 1.751 | no |

At 1.0, the add/upsert and dimensions questions would be rejected even though the answers are in the docs. At **1.3**, every real question gets through and every off-topic one is still blocked.

**2. Confidence.** This uses the best kept distance: `high` < 0.5, `medium` < 1.0, `low` otherwise. Because the threshold is 1.3, `low` means a best match between 1.0 and 1.3. None of my test questions scored below 0.5, so `high` never came up. The closest was 0.541.

**3. System prompt.** The model must answer only from the context and never make up information. It must reply "I don't know." when the context doesn't cover the question, answer only the covered part of a partial question, and cite every fact as `(file.md)`. Retrieved text is treated as data, not instructions, which carries over from the prompt injection exercise.

**4. Structured response.** Every call returns:

```python
{"answer": str, "sources": list[str], "confidence": str, "chunks_retrieved": int}
```

`sources` holds the unique filenames of the kept chunks, in order. If the model refuses, `sources` is set to `[]`, because the answer didn't actually use them.

## Test results

The full run output is in `test_output.txt`.

| Type | Question | Distances (kept) | Confidence | Sources | Answer |
|---|---|---|---|---|---|
| In-scope | What embedding model does ChromaDB use by default? | 0.541, 0.683, 0.891 (3) | medium | chromadb.md, embeddings.md | "ChromaDB uses the all-MiniLM-L6-v2 sentence-transformer model by default. (embeddings.md)" |
| Partial | How do I choose a good chunk overlap for PDFs? | 0.883, 0.972, 1.073 (3) | medium | [] | "I don't know." |
| Out-of-scope | Who won the 2018 World Cup? | 1.751, 1.780, 1.791 (0) | low | [] | No-info response, LLM not called |
| Ambiguous | What is a good threshold? | 1.147, 1.277, 1.281 (3) | low | [] | "I don't know." |

Observations:

- The in-scope answer is correct and cited. The `my_rag.py` version invented details about `add`. This one stuck to the docs.
- The out-of-scope query was stopped by the threshold, so it never reached the model.
- The partial and ambiguous queries refused completely. The docs do have related material: chunking.txt gives a 10-20% overlap, and semantic_search.md mentions a cutoff around 1.0 that "has to be tuned". So the 1b model at temperature 0 leans towards over-refusing instead of answering the part it can. That's the safer way to fail, but a bigger model would probably handle the "answer what you can" rule better.
