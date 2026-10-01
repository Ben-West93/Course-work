# Module 07 — Prompt Workshop

Three "bad vs. good" prompt pairs sent to an LLM and printed side by side, plus a system prompt for a Course Study Assistant. Practice for the Module 8 RAG system prompt.

## Run

```bash
cd ~/Course-work
.venv/bin/python module_07/prompt_workshop/prompt_workshop.py
.venv/bin/python module_07/prompt_workshop/test_prompt_workshop.py   # 11 offline tests
```

No API key is needed. The script uses `real_llm()` (OpenAI `gpt-4o-mini`) only when `OPENAI_API_KEY` is set **and** the `openai` package is installed (`.venv/bin/python -m pip install openai`). Otherwise it uses `mock_llm()`, which returns typical canned replies for vague and engineered prompts. The first line of output shows which one is active.

## Techniques

| Technique | What it does | Used in |
|---|---|---|
| Role framing | Sets the model's expertise, audience and tone ("You are an expert Python instructor…") | Task 1, Task 3 system prompt |
| Specificity | Sets the exact audience, structure, length, schema and allowed values | Tasks 1, 2, 3 |
| Few-shot | Shows one example of the output shape you want | Task 2 (one sample JSON object) |
| Output constraints | Fixes the format ("Respond ONLY with the raw JSON array", "Source: <doc>") | Task 2, Task 3 |

## Output comparison

| Task | Bad prompt → response | Good prompt → response |
|---|---|---|
| 1. Explain `st.session_state` | "what is session state in streamlit" → vague, no example | Role + beginner audience + 3-part structure → why it exists, a counter example, one use case |
| 2. Task list → JSON | "turn this into json" → "Sure! Here's…" + ```` ```json ```` fence + wrong shape; **fails** `json.loads` | Schema + allowed values + example + "raw JSON only" → 4 objects with `title`/`priority`/`status`; **passes** `json.loads` |
| 3. Study Assistant | "help with course questions" → asks a follow-up question, can't answer | System prompt + labelled context + question → grounded 3-sentence answer ending `Source: chunking_strategies.md` |

The script checks Task 2 by running `json.loads` on the raw response, with no cleanup, and prints PASS/FAIL for each prompt.

## Study Assistant system prompt (summary)

1. Role: "You are an AI Course Assistant…"
2. Answer **only** from the provided context documents.
3. If the answer isn't there, say exactly *"I don't know based on the provided course notes."* Don't make anything up.
4. Keep answers under 150 words.
5. End every answer with `Source: <document name>`.

The full text is printed at the end of each run.

## Edge cases handled

- `OPENAI_API_KEY` missing or blank → uses `mock_llm`.
- Key set but `openai` not installed → uses `mock_llm` and says why.
- Real API call fails (network, auth, rate limit) → shows the error type and uses the mock reply, so the run finishes.
- Task 2 parser rejects markdown fences, prose, a JSON object instead of an array, and objects with missing or extra keys.
