"""Tests for the static analyzer behind `aura-state check`.

Each uses an adversarial design and asserts the verdict (per CLAUDE.md rule 8).
"""
from aura_state.check import check_flow


SAFE_SQL = {
    "name": "sql-agent", "entry": "Ask",
    "edges": [["Ask", "GenSQL"], ["GenSQL", "Validate"], ["Validate", "Execute"]],
    "nodes": [
        {"id": "Ask", "kind": "extract", "capability": "untrusted", "obligations": []},
        {"id": "GenSQL", "kind": "extract", "capability": "plain", "obligations": ["read_only == True"]},
        {"id": "Validate", "kind": "sanitizer", "capability": "sanitizer", "obligations": []},
        {"id": "Execute", "kind": "tool", "tool_name": "db.query", "side_effect": "external", "obligations": []},
    ],
}

NAIVE_SQL = {
    "name": "sql-naive", "entry": "Ask",
    "edges": [["Ask", "Execute"]],
    "nodes": [
        {"id": "Ask", "kind": "extract", "capability": "untrusted", "obligations": []},
        {"id": "Execute", "kind": "tool", "tool_name": "db.query", "side_effect": "external", "obligations": []},
    ],
}


def test_validated_sql_agent_is_proven():
    r = check_flow(SAFE_SQL)
    assert r.verified is True
    assert not [f for f in r.findings if f.check == "taint"]


def test_naive_agent_flags_injection_path():
    r = check_flow(NAIVE_SQL)
    assert r.verified is False
    taint = [f for f in r.findings if f.check == "taint"]
    assert taint and taint[0].node == "Execute"


def test_tool_side_effect_read_is_not_a_sink():
    # a read-only tool reached from untrusted input is NOT an injection sink
    flow = {"name": "reader", "entry": "Ask", "edges": [["Ask", "Search"]],
            "nodes": [{"id": "Ask", "kind": "extract", "capability": "untrusted"},
                      {"id": "Search", "kind": "tool", "tool_name": "vectordb.search", "side_effect": "read"}]}
    r = check_flow(flow)
    assert not [f for f in r.findings if f.check == "taint"]


def test_flags_hardcoded_secret_in_prompt():
    flow = {"name": "leaky", "entry": "A", "edges": [],
            "nodes": [{"id": "A", "kind": "extract", "capability": "plain",
                       "system_prompt": "call with api_key=sk-abcdefghij1234567890ABCD"}]}
    r = check_flow(flow)
    assert any(f.check == "policy" for f in r.findings)


def test_flags_contradictory_obligations():
    flow = {"name": "bad", "entry": "A", "edges": [],
            "nodes": [{"id": "A", "kind": "extract", "capability": "plain",
                       "obligations": ["x > 5", "x < 3"]}]}
    r = check_flow(flow)
    assert r.verified is False
    assert any(f.check == "obligation" for f in r.findings)


def test_empty_flow_is_not_verified():
    assert check_flow({"name": "empty", "nodes": []}).verified is False
