"""
L7 — Prompt Workshop
====================
Run with:
    python prompt_workshop.py

Sends three bad/good prompt pairs to an LLM and prints the results side by
side, then prints the Course Study Assistant system prompt.

Uses real_llm() (OpenAI gpt-4o-mini) when OPENAI_API_KEY is set and the
openai package is installed; otherwise falls back to mock_llm() so the script
always runs with zero API dependencies.
"""

import json
import os

# ── Mock LLM ─────────────────────────────────────────────────────────────────
# Canned replies that imitate how a real model typically responds to a vague
# prompt vs. an engineered one, so the comparison is meaningful offline.
MOCK_REPLIES = {
    "session_state": {
        "bad": (
            "Session state is a feature in Streamlit that lets you store "
            "things. It's useful for state. There are many ways to use it "
            "depending on your app."
        ),
        "good": (
            "Streamlit re-runs your whole script top to bottom on every "
            "interaction, so normal variables reset each time. "
            "st.session_state is a dictionary-like object that survives those "
            "re-runs for one user's browser session.\n\n"
            "Example:\n"
            "    if \"count\" not in st.session_state:\n"
            "        st.session_state.count = 0\n"
            "    if st.button(\"Add one\"):\n"
            "        st.session_state.count += 1\n"
            "    st.write(st.session_state.count)\n\n"
            "Key use case: remembering values between clicks, such as a "
            "counter, a logged-in user, or chat history."
        ),
    },
    "json": {
        "bad": (
            "Sure! Here's your task list converted to JSON:\n\n"
            "```json\n"
            "{\"tasks\": [\"finish the module 7 exercises\", "
            "\"review chunking strategies notes\", "
            "\"watch the ChromaDB guided example video\", "
            "\"start the module project\"]}\n"
            "```\n\n"
            "Let me know if you'd like me to add any other fields!"
        ),
        "good": json.dumps([
            {"title": "Finish the module 7 exercises", "priority": "high", "status": "todo"},
            {"title": "Review chunking strategies notes", "priority": "medium", "status": "todo"},
            {"title": "Watch the ChromaDB guided example video", "priority": "medium", "status": "todo"},
            {"title": "Start the module project", "priority": "high", "status": "todo"},
        ], indent=2),
    },
    "course": {
        "bad": (
            "Of course! I'd be happy to help with your course questions. "
            "What course are you taking, and what would you like to know?"
        ),
        "good": (
            "Chunk overlap repeats a few sentences at the edge of each chunk "
            "so that an idea split across a boundary still appears whole in "
            "at least one chunk. This keeps retrieval from missing context "
            "that sits between two chunks. The notes suggest an overlap of "
            "roughly 10–20% of the chunk size.\n\n"
            "Source: chunking_strategies.md"
        ),
    },
}


def mock_llm(prompt: str, system: str = "") -> str:
    """Return a canned reply so the script runs without an API key."""
    text = prompt.lower()
    if "session" in text:
        task = "session_state"
    elif "json" in text:
        task = "json"
    elif "course" in text:
        task = "course"
    else:
        return f"[MOCK RESPONSE]\nPrompt length: {len(prompt)} chars\nFirst 80 chars of prompt: {prompt[:80]!r}"

    # Treat a prompt as "engineered" if it carries a system prompt, a role
    # frame, or an explicit output constraint.
    engineered = bool(system) or text.startswith("you are") or "respond only" in text
    return "[MOCK] " + MOCK_REPLIES[task]["good" if engineered else "bad"]


# ── Real LLM (optional) ───────────────────────────────────────────────────────
def real_llm(prompt: str, system: str = "") -> str:
    """Call OpenAI with the given prompt. Requires OPENAI_API_KEY env var."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content


def choose_llm():
    """Use real_llm only when a non-blank key is set and openai is installed."""
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        return mock_llm, "mock_llm (OPENAI_API_KEY not set)"
    try:
        import openai  # noqa: F401
    except ImportError:
        return mock_llm, "mock_llm (OPENAI_API_KEY set, but openai package not installed)"
    return real_llm, "real_llm (gpt-4o-mini)"


# Choose which function to use throughout this exercise
llm, LLM_LABEL = choose_llm()


def ask(prompt: str, system: str = "") -> str:
    """Call the active LLM; if a real API call fails, fall back to the mock."""
    try:
        return llm(prompt, system)
    except Exception as exc:  # network, auth, rate-limit errors, etc.
        return f"[API error: {exc.__class__.__name__} — falling back to mock]\n" + mock_llm(prompt, system)


# ── Helper ────────────────────────────────────────────────────────────────────
def compare(task_name: str, bad_prompt: str, good_prompt: str, good_system: str = ""):
    """Run both prompts and print results side by side. Returns both responses."""
    bad_response = ask(bad_prompt)
    good_response = ask(good_prompt, good_system)
    print(f"\n{'='*60}")
    print(f"TASK: {task_name}")
    print(f"\n--- BAD PROMPT ---\n{bad_prompt}\n")
    print(f"BAD RESPONSE:\n{bad_response}")
    print(f"\n--- GOOD PROMPT ---")
    if good_system:
        print("(sent with STUDY_ASSISTANT_SYSTEM_PROMPT as the system message)")
    print(f"{good_prompt}\n")
    print(f"GOOD RESPONSE:\n{good_response}")
    return bad_response, good_response


def parse_task_json(response: str):
    """Strictly parse a Task 2 response: must be a raw JSON array of task objects.

    Returns (ok, message). No markdown stripping — the output constraint in the
    good prompt is what should make the raw response parseable.
    """
    text = response.removeprefix("[MOCK] ").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return False, f"not valid JSON ({exc.msg} at char {exc.pos})"
    if not isinstance(data, list):
        return False, f"valid JSON but a {type(data).__name__}, not an array"
    required = {"title", "priority", "status"}
    for i, item in enumerate(data):
        if not isinstance(item, dict) or set(item) != required:
            return False, f"item {i} does not have exactly the keys {sorted(required)}"
    return True, f"valid JSON array of {len(data)} objects with keys {sorted(required)}"


# ── Task 1 — Code explanation ─────────────────────────────────────────────────
# Vague: no audience, no format, no length.
bad_prompt_1 = "what is session state in streamlit"

# Techniques: ROLE FRAMING ("expert Python instructor") sets tone and expertise;
# SPECIFICITY pins the audience (beginner), the exact structure (why it exists,
# one code example, one use case), and a length limit.
good_prompt_1 = (
    "You are an expert Python instructor who teaches Streamlit to beginners.\n"
    "Explain what st.session_state does in Streamlit to a student who knows "
    "basic Python but has never built a Streamlit app.\n"
    "Structure your answer as:\n"
    "1. One or two sentences on why it exists (Streamlit re-runs the script "
    "on every interaction).\n"
    "2. A short, runnable code example (under 10 lines) using a button counter.\n"
    "3. One key real-world use case.\n"
    "Keep the whole answer under 150 words and avoid jargon."
)


# ── Task 2 — Data formatting ──────────────────────────────────────────────────
TASK_LIST = """
finish the module 7 exercises
review chunking strategies notes
watch the ChromaDB guided example video
start the module project
"""

# Vague: no schema, no format rules — invites prose and ```json fences.
bad_prompt_2 = f"turn this into json\n{TASK_LIST}"

# Techniques: SPECIFICITY (exact schema, allowed values, one object per line
# of input); FEW-SHOT (one example object showing the exact shape);
# OUTPUT CONSTRAINTS (raw JSON array only — no markdown fences, no prose) so
# the response can go straight into json.loads().
good_prompt_2 = (
    "Convert the task list below into a JSON array.\n\n"
    "Rules:\n"
    "- One object per task, in the same order as the list.\n"
    "- Each object has exactly these keys: \"title\", \"priority\", \"status\".\n"
    "- \"title\": the task rewritten in sentence case.\n"
    "- \"priority\": one of \"high\", \"medium\", \"low\" (exercises and the "
    "module project are \"high\"; review and videos are \"medium\").\n"
    "- \"status\": one of \"todo\", \"in_progress\", \"done\" (use \"todo\" "
    "unless the task says otherwise).\n\n"
    "Example of one object:\n"
    "{\"title\": \"Read the embeddings chapter\", \"priority\": \"medium\", \"status\": \"todo\"}\n\n"
    "Respond ONLY with the raw JSON array. No markdown, no code fences, "
    "no explanation before or after.\n\n"
    f"Task list:{TASK_LIST}"
)


# ── Task 3 — System prompt design ────────────────────────────────────────────
# Vague: no role, no context, no constraints — the model can only ask back.
bad_prompt_3 = "help with course questions"

# Techniques: ROLE FRAMING (clear role), SPECIFICITY (grounding, length, and
# citation rules), and OUTPUT CONSTRAINTS (fixed answer format and a fixed
# fallback sentence so "I don't know" answers are consistent and testable).
STUDY_ASSISTANT_SYSTEM_PROMPT = """\
You are an AI Course Assistant for a Python and applied AI course. You help
students understand course material using the course notes you are given.

Rules:
1. Answer ONLY using the context documents provided in the user message.
   Do not use outside knowledge, even if you know the answer.
2. If the context does not contain the answer, reply exactly:
   "I don't know based on the provided course notes." Never guess or make
   up information.
3. Keep every answer under 150 words.
4. End every answer with a line naming the source document(s) you used,
   in the form: Source: <document name>
5. Use plain language suitable for a beginner. If the question is ambiguous,
   answer the most likely reading and say which one you chose."""

# Techniques: CONTEXT INJECTION (labelled source snippets the system prompt
# can cite), SPECIFICITY (one clear question), and OUTPUT CONSTRAINTS
# (answer-then-source format, word limit restated).
good_prompt_3 = (
    "Context documents:\n\n"
    "[chunking_strategies.md]\n"
    "Chunk overlap repeats a small amount of text at the boundary of each "
    "chunk so ideas split across two chunks are not lost. An overlap of "
    "10-20% of the chunk size is a common starting point.\n\n"
    "[embeddings_intro.md]\n"
    "Embeddings turn text into vectors so that texts with similar meaning "
    "end up close together. Similarity is usually measured with cosine "
    "similarity.\n\n"
    "Question: Why do we use overlap when chunking course notes, and how much "
    "overlap should I start with?\n\n"
    "Answer in 2-4 sentences, then the Source line."
)


def main():
    print(f"Using: {LLM_LABEL}")

    compare("Code Explanation — st.session_state", bad_prompt_1, good_prompt_1)

    bad_2, good_2 = compare("Data Formatting — task list to JSON", bad_prompt_2, good_prompt_2)
    print("\n--- JSON CHECK (json.loads on the raw response) ---")
    for label, response in (("bad ", bad_2), ("good", good_2)):
        ok, message = parse_task_json(response)
        print(f"{label}: {'PASS' if ok else 'FAIL'} — {message}")

    compare(
        "System Prompt Design — Study Assistant",
        bad_prompt_3,
        good_prompt_3,
        good_system=STUDY_ASSISTANT_SYSTEM_PROMPT,
    )

    # Print the final system prompt
    print(f"\n{'='*60}")
    print("STUDY ASSISTANT SYSTEM PROMPT:")
    print(STUDY_ASSISTANT_SYSTEM_PROMPT)


if __name__ == "__main__":
    main()
