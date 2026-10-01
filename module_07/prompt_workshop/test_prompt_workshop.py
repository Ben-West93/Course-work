"""
Tests for prompt_workshop.py — run without any API key.

Run with:
    python test_prompt_workshop.py
or (if pytest is installed):
    pytest test_prompt_workshop.py -v
"""

import contextlib
import io
import json
import os
import sys

os.environ.pop("OPENAI_API_KEY", None)  # guarantee mock mode before import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prompt_workshop as pw


def run_main():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pw.main()
    return buf.getvalue()


def test_defaults_to_mock_without_key():
    assert pw.llm is pw.mock_llm
    assert "not set" in pw.LLM_LABEL


def test_blank_key_still_uses_mock():
    os.environ["OPENAI_API_KEY"] = "   "
    try:
        llm, _ = pw.choose_llm()
        assert llm is pw.mock_llm
    finally:
        os.environ.pop("OPENAI_API_KEY")


def test_key_without_openai_package_uses_mock():
    os.environ["OPENAI_API_KEY"] = "sk-test-placeholder"
    saved = sys.modules.get("openai")
    sys.modules["openai"] = None  # makes `import openai` raise ImportError
    try:
        llm, label = pw.choose_llm()
        assert llm is pw.mock_llm and "not installed" in label
    finally:
        os.environ.pop("OPENAI_API_KEY")
        if saved is None:
            sys.modules.pop("openai")
        else:
            sys.modules["openai"] = saved


def test_api_error_falls_back_to_mock():
    def broken_llm(prompt, system=""):
        raise ConnectionError("no network")
    original = pw.llm
    pw.llm = broken_llm
    try:
        out = pw.ask(pw.good_prompt_1)
        assert "API error: ConnectionError" in out and "[MOCK]" in out
    finally:
        pw.llm = original


def test_no_todo_placeholders_left():
    for name in ("bad_prompt_1", "good_prompt_1", "bad_prompt_2", "good_prompt_2",
                 "bad_prompt_3", "good_prompt_3", "STUDY_ASSISTANT_SYSTEM_PROMPT"):
        value = getattr(pw, name)
        assert value.strip() and "TODO" not in value, name


def test_good_json_is_raw_and_valid():
    response = pw.ask(pw.good_prompt_2)
    ok, message = pw.parse_task_json(response)
    assert ok, message
    body = response.removeprefix("[MOCK] ")
    assert "```" not in body and not body.lower().startswith("sure")
    titles = [item["title"].lower() for item in json.loads(body)]
    expected = [line.strip().lower() for line in pw.TASK_LIST.strip().splitlines()]
    assert titles == expected


def test_bad_json_is_rejected():
    ok, _ = pw.parse_task_json(pw.ask(pw.bad_prompt_2))
    assert not ok


def test_parser_rejects_wrong_shapes():
    assert not pw.parse_task_json('{"title": "x", "priority": "high", "status": "todo"}')[0]
    assert not pw.parse_task_json('[{"title": "x", "priority": "high"}]')[0]
    assert pw.parse_task_json("[]")[0]


def test_system_prompt_covers_all_rules():
    sp = pw.STUDY_ASSISTANT_SYSTEM_PROMPT
    assert sp.startswith("You are an AI Course Assistant")
    assert "ONLY using the context" in sp
    assert "I don't know" in sp
    assert "150 words" in sp
    assert "Source:" in sp


def test_good_prompt_3_sends_system_prompt_and_cites_source():
    response = pw.ask(pw.good_prompt_3, pw.STUDY_ASSISTANT_SYSTEM_PROMPT)
    assert "Source: chunking_strategies.md" in response
    assert len(response.split()) < 150


def test_main_prints_three_tasks_and_ends_with_system_prompt():
    out = run_main()
    assert out.count("TASK: ") == 3
    assert out.count("=" * 60) == 4
    assert "good: PASS" in out and "bad : FAIL" in out
    assert out.rstrip().endswith(pw.STUDY_ASSISTANT_SYSTEM_PROMPT.rstrip())


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
