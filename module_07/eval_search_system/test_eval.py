"""
Multi-angle tests for my_eval.py
================================
Run with:
    python test_eval.py

Checks the zero-division guards, threshold boundary, broad vs specific
queries, n_results clamping, unindexed-ID rejection, and that the printed
averages match a manual recomputation for every benchmark setting.
"""

import contextlib
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import my_eval as m  # noqa: E402  (needs the path above)


def run(**kw):
    """Call evaluate() with its output captured; return (averages, text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ret = m.evaluate(**kw)
    return ret, buf.getvalue()


def test_zero_retrieval():
    # Thresholds this strict let nothing through, so len(retrieved) == 0.
    for t in (0.0, 0.1):
        (p, r), out = run(n_results=5, distance_threshold=t)
        print(f"  t={t}: {out.strip().splitlines()[-1].strip()}")
        assert (p, r) == (0.0, 0.0)


def test_threshold_boundary():
    # A hit whose distance equals the threshold is kept; just below drops it.
    saved = m.eval_set
    m.eval_set = [saved[0]]
    try:
        q = saved[0]["query"]
        d = m.collection.query(query_texts=[q], n_results=1)["distances"][0][0]
        (p_on, _), _ = run(n_results=1, distance_threshold=d)
        (p_below, _), _ = run(n_results=1, distance_threshold=d - 1e-6)
        print(f"  t=d ({d:.4f}): P={p_on}   t=d-1e-6: P={p_below}")
        assert p_on == 1.0 and p_below == 0.0
    finally:
        m.eval_set = saved


def test_broad_vs_specific():
    # Broad query: many relevant docs, so high precision but low recall.
    # Specific identifiers: the right doc ranks first but scores > 0.5.
    saved = m.eval_set
    m.eval_set = [
        {"query": "web development", "relevant_ids": list(m.ids)},
        {"query": "grid-template-columns", "relevant_ids": ["doc_14"]},
        {"query": "st.cache_data", "relevant_ids": ["doc_07"]},
        {"query": "422 error Pydantic", "relevant_ids": ["doc_01"]},
    ]
    try:
        _, out = run(n_results=3)
        print(out.rstrip())
        _, out_t = run(n_results=3, distance_threshold=0.5)
        print(out_t.rstrip())
        for entry in m.eval_set[1:]:
            top = m.collection.query(query_texts=[entry["query"]], n_results=1)
            assert top["ids"][0][0] == entry["relevant_ids"][0]
    finally:
        m.eval_set = saved


def test_n_results_clamped():
    (_, r), out = run(n_results=50)
    print(f"  n_results=50: {out.strip().splitlines()[-1].strip()}")
    assert r == 1.0


def test_unindexed_id_rejected():
    with open(os.path.join(HERE, "my_eval.py")) as f:
        src = f.read().replace('"doc_12"]}', '"doc_99"]}')
    try:
        exec(compile(src, "my_eval_patched", "exec"), {"__name__": "patched"})
    except AssertionError as e:
        print(f"  caught: {e}")
        return
    raise AssertionError("unindexed ID was not rejected")


def test_averages_match():
    settings = [
        dict(n_results=3),
        dict(n_results=5),
        dict(n_results=5, distance_threshold=0.5),
        dict(n_results=5, distance_threshold=0.7),
    ]
    for kw in settings:
        (p, r), out = run(**kw)
        ps = [float(x) for x in re.findall(r"Query \d+: P=([\d.]+)%", out)]
        rs = [float(x) for x in re.findall(r"R=([\d.]+)%  \|", out)]
        printed = re.search(r"AVERAGE: P=([\d.]+)%  R=([\d.]+)%", out).groups()
        # Per-query lines are rounded to 0.1%, so allow a tiny tolerance.
        assert abs(sum(ps) / len(ps) - p * 100) < 0.06
        assert abs(sum(rs) / len(rs) - r * 100) < 0.06
        assert (f"{p*100:.1f}", f"{r*100:.1f}") == printed
        print(f"  {kw}: P={printed[0]}% R={printed[1]}%  MATCH")
    assert run(n_results=3)[0] == run(n_results=3)[0]
    print("  repeat run identical")


if __name__ == "__main__":
    tests = [
        ("A. Strict threshold, 0 docs pass", test_zero_retrieval),
        ("B. Threshold boundary", test_threshold_boundary),
        ("C. Broad vs specific keyword queries", test_broad_vs_specific),
        ("D. n_results larger than collection", test_n_results_clamped),
        ("E. Unindexed ID rejected at load", test_unindexed_id_rejected),
        ("F. Averages match manual recomputation", test_averages_match),
    ]
    for name, fn in tests:
        print(f"--- {name} ---")
        fn()
    print(f"\nALL {len(tests)} TESTS PASSED")
