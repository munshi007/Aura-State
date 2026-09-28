"""0021: capability containment / least-privilege — the no-attacker over-reach class."""
from aura_state.check import check_flow
from aura_state.verification.capability_containment import analyze_containment


def _flow(manifest=None, extra_nodes=None, edges=None):
    nodes = [{"id": "Ask", "kind": "extract", "capability": "plain"},
             {"id": "Pay", "kind": "tool", "tool_name": "payment.charge", "side_effect": "external"}]
    f = {"name": "a", "entry": "Ask", "edges": edges or [["Ask", "Pay"]], "nodes": nodes + (extra_nodes or [])}
    if manifest is not None:
        f["manifest"] = manifest
    return f


def test_reachable_over_reach_is_flagged_fixes_0021():
    # declared read-only, but a reachable external payment -> over-privileged action
    r = check_flow(_flow({"side_effects": ["read"]}))
    assert r.verified is False
    assert r.summary["least_privilege"] == "exceeded"
    assert any(f.check == "least-privilege" and f.node == "Pay" for f in r.findings)


def test_no_manifest_is_not_declared_no_finding_fixes_0021():
    r = check_flow(_flow(manifest=None))
    assert r.summary["least_privilege"] == "not declared"
    assert not any(f.check == "least-privilege" for f in r.findings)


def test_within_scope_is_contained_fixes_0021():
    f = {"name": "c", "entry": "Ask", "edges": [["Ask", "Read"]], "manifest": {"side_effects": ["read"]},
         "nodes": [{"id": "Ask", "kind": "extract"},
                   {"id": "Read", "kind": "tool", "tool_name": "db.read", "side_effect": "read"}]}
    r = check_flow(f)
    assert r.summary["least_privilege"] == "contained"
    assert not any(f.check == "least-privilege" for f in r.findings)


def test_unreachable_effect_is_not_flagged_fixes_0021():
    # an out-of-scope tool that the entry can NEVER reach is not a finding
    orphan = [{"id": "Del", "kind": "tool", "tool_name": "db.delete", "side_effect": "write"}]
    f = {"name": "o", "entry": "Ask", "edges": [["Ask", "Read"]], "manifest": {"side_effects": ["read"]},
         "nodes": [{"id": "Ask", "kind": "extract"},
                   {"id": "Read", "kind": "tool", "tool_name": "db.read", "side_effect": "read"}] + orphan}
    assert check_flow(f).summary["least_privilege"] == "contained"


def test_tool_and_data_class_allowlists_fixes_0021():
    res = analyze_containment(
        [{"id": "T", "kind": "tool", "tool_name": "shell.exec", "side_effect": "external", "data_class": "private"}],
        [], "T",
        {"tools": ["db.read"], "data_classes": ["public"]})
    assert res.verified is False
    dims = {d for _, d, _ in res.violations}
    assert "tool" in dims and "data_class" in dims


def test_empty_manifest_is_not_declared_fixes_0021():
    res = analyze_containment([{"id": "T", "kind": "tool", "tool_name": "x", "side_effect": "external"}], [], "T", {})
    assert res.declared is False and res.verified is True
