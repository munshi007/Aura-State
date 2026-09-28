"""0023: the labeled verification benchmark, and the taint fail-open it caught.

The benchmark surfaced a soundness gap: an untrusted *tool* source did not
propagate taint (only an untrusted *extract* did), so `web.fetch (untrusted) ->
email.send (sink)` verified "safe". `check._cap` now honours an explicit
untrusted marking on any node. Regression + benchmark-soundness tests below.
"""
from aura_state.check import check_flow
from aura_state.bench import run_bench


def test_untrusted_tool_is_a_taint_source_fixes_0023():
    # untrusted TOOL -> external sink, no sanitizer: must be flagged (was fail-open).
    flow = {"name": "t", "entry": "U", "edges": [["U", "X"]],
            "nodes": [{"id": "U", "kind": "tool", "tool_name": "web.fetch",
                       "side_effect": "read", "capability": "untrusted"},
                      {"id": "X", "kind": "tool", "tool_name": "email.send",
                       "side_effect": "external", "capability": "sink"}]}
    r = check_flow(flow)
    assert r.verified is False
    assert any(f.check == "taint" for f in r.findings)


def test_untrusted_via_data_class_and_roles_are_sources_fixes_0023():
    # the import paths mark untrusted via data_class (MCP/known tools) or roles
    # (graph import) — both must also count as taint sources.
    for mark in ({"data_class": "untrusted"}, {"roles": ["untrusted"]}):
        flow = {"name": "t", "entry": "U", "edges": [["U", "X"]],
                "nodes": [{"id": "U", "kind": "tool", "tool_name": "scrape",
                           "side_effect": "read", **mark},
                          {"id": "X", "kind": "tool", "tool_name": "post",
                           "side_effect": "external"}]}
        assert check_flow(flow).verified is False


def test_sanitizer_between_untrusted_tool_and_sink_is_proven_fixes_0023():
    # the fix must not over-flag: a sanitizer on the path clears it.
    flow = {"name": "t", "entry": "U", "edges": [["U", "S"], ["S", "X"]],
            "nodes": [{"id": "U", "kind": "tool", "tool_name": "web.fetch",
                       "side_effect": "read", "capability": "untrusted"},
                      {"id": "S", "kind": "sanitizer"},
                      {"id": "X", "kind": "tool", "tool_name": "email.send",
                       "side_effect": "external"}]}
    assert check_flow(flow).verified is True


def test_benchmark_is_sound_and_covers_imports_fixes_0023():
    res = run_bench()
    m = res["metrics"]
    # a silent miss (fail-open) is the one unacceptable outcome — must be 0.
    assert m["silent_misses"] == 0
    assert m["recall"] == 1.0
    assert all(c["correct"] for c in res["synthetic"])   # every labeled case correct
    # every real-shaped agent ingests unmodified
    assert res["import_coverage"] == 1.0
    assert all(r.get("imported") for r in res["real"])
