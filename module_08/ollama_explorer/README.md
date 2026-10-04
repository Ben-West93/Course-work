# Ollama Explorer

Module 08 exercise. Runs four small experiments against a local model through Ollama's `/api/chat` endpoint to see how system prompts, context, input length and temperature change the output.

## Setup

Install Ollama (macOS: `brew install ollama`), then in one terminal:

```bash
ollama serve
```

and in another:

```bash
ollama pull llama3.2:1b
```

The script only needs `requests`. Change `MODEL` at the top of the file if you pulled something else.

## Running

From the repo root:

```bash
.venv/bin/python module_08/ollama_explorer/ollama_explorer.py
```

If Ollama isn't running, every call prints a connection error with 0.00s and the script still runs to the end. A missing model shows up as an HTTP 404 error message instead of a crash.

## Files

- `ollama_explorer.py` - finished script
- `starter.py` - original starter code from the course

## Experiments

Results below are from llama3.2:1b on an Apple Silicon Mac.

**1. System prompts** - "What is an API?" with no system prompt, an ELI5 prompt, and a senior-architect prompt.

| Prompt | Words | Time |
|---|---|---|
| None | 382 | 3.70s |
| ELI5 | 246 | 1.74s |
| Architect | 355 | 2.75s |

ELI5 switched to a toy-box analogy with no jargon. The architect version added REST/SOAP/gRPC and HTTP methods but opened almost the same way as the default answer.

**2. Context grounding** - a short paragraph about the Golden Gate Bridge plus a strict "answer only from context" rule.

- "What year did it open?" -> `May 27, 1937.`
- "How much did it cost?" -> gave the exact refusal sentence, then added a line explaining the cost wasn't in the context.

So the model stayed grounded, but it doesn't always say *only* the refusal, which matters if you're matching on the exact string.

**3. Timing** - short (5 words), medium (21) and long (56) questions.

| Length | Words in | Words out | Time |
|---|---|---|---|
| Short | 5 | 227 | 2.16s |
| Medium | 21 | 563 | 3.86s |
| Long | 56 | 557 | 5.03s |

**4. Temperature** - "Tell me a one-sentence fact about the ocean." three times each at 0.1 and 1.0. At 0.1, two runs were identical and the third only changed the ending. At 1.0 every run gave a different fact.

## Takeaways

- Response time depends mostly on how long the answer is. Input length barely moved it on a 1B model.
- Short grounded answers (experiment 2) come back in well under a second, so keeping RAG answers tight helps latency a lot.
- System prompts are the easiest way to control tone and depth.
- Use low temperature for factual or RAG answers where you want repeatable output.
- Small models follow grounding rules here but tack on extra text, so check refusals with "contains" instead of exact equality.
- The first call after `ollama serve` is slower because the model has to load into memory. The script uses a 120s timeout for that.
