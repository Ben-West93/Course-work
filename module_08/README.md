# Module 8: RAG Prompt Builder

The prompt assembly step of a RAG pipeline. It takes a question and a list of
simulated retrieved chunks, builds the full prompt, and prints it with a token
estimate. The full pipeline diagram is in [rag_architecture.md](rag_architecture.md).

- `prompt_builder.py`: the finished script
- `starter.py`: the original starter file with the TODOs

## Running

From the repo root:

```bash
.venv/bin/python module_08/prompt_builder.py
```

No dependencies outside the standard library.

## Prompt assembly pattern

```
<SYSTEM_PROMPT>

CONTEXT:
[Source: doc1.txt]
<chunk text>

---

[Source: doc2.txt]
<chunk text>

USER QUESTION: <question>
```

The system prompt tells the model to:
1. answer only from the context
2. reply "I don't have enough information to answer that." when the answer isn't there
3. cite the source filename for each fact

Chunks are joined with `"\n\n---\n\n"` so the model can see where one chunk ends
and the next begins. Each chunk has its source label, so the model can cite it.

Edge cases handled:
- empty chunk list: context becomes `(no documents retrieved)`, so the layout stays the same and the model should give the fallback answer
- chunk missing `source`: labelled `[Source: unknown]` instead of raising `KeyError`
- chunk missing `text`: treated as an empty string
- blank question: stripped, and the prompt still builds

## Token estimate

```
tokens ≈ len(text) // 4
```

Roughly 4 characters per token for English text. `len()` counts Unicode code
points, so text with lots of CJK or emoji will be under-estimated, since those
usually take more than one token per character. Text shorter than 4 characters
comes out as 0.

## Test results

| Case | Chunks | Chars | Context tokens | Total tokens |
|---|---|---|---|---|
| Refund window | 3 (2 sources) | 839 | ~99 | ~209 |
| Embedding model | 2 (2 sources) | 736 | ~67 | ~184 |
| Empty chunk list | 0 | 448 | ~0 | ~112 |
| Missing `source` key | 1 | 497 | ~18 | ~124 |
| Blank question + Unicode | 1 | 459 | ~16 | ~114 |

The system prompt plus the labels cost about 110 tokens on every request, even
when there are no chunks. In the short tests that fixed overhead is more than
half the prompt.

Scaling with one large chunk:

| Repeat | Chars | Tokens |
|---|---|---|
| 1 | 497 | ~124 |
| 10 | 1,047 | ~261 |
| 100 | 6,538 | ~1,634 |
| 1,000 | 61,439 | ~15,359 |

The estimate grows linearly with context size. Once there's a lot of context the
fixed overhead hardly matters, and retrieved text is nearly all of the cost. That
is why top-k and chunk size affect cost more than the system prompt does.
