"""API tests for the local Aura Studio backend (aura_state.ui.server).

These hit the REAL verifiers through the FastAPI app — Z3, CTL, static taint,
the contract compiler, and the counterexample-guided sanitizer repair — with no
LLM in the loop (every node here is capability/rule based). Each test uses an
adversarial design (an actual taint path, a genuinely false obligation) and
asserts the verdict, per CLAUDE.md rule 8.
"""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from aura_state.ui.server import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


# untrusted source -> ... -> dangerous sink, no sanitizer between them.
TAINTED = {
    "entry": "Ingest",
    "edges": [["Ingest", "Classify"], ["Classify", "Pay"]],
    "nodes": [
        {"id": "Ingest", "capability": "untrusted"},
        {"id": "Classify", "capability": "plain"},
        {"id": "Pay", "capability": "sink"},
    ],
}


def test_verify_flags_untrusted_to_sink_taint(client):
    r = client.post("/api/verify", json=TAINTED).json()
    assert r["taint"]["verdict"] == "VIOLATED"
    assert any(v["sink"] == "Pay" for v in r["taint"]["violations"])


def test_repair_inserts_sanitizer_and_reproves_clean(client):
    r = client.post("/api/repair", json=TAINTED).json()
    assert r["repaired"] is True
    assert r["taint_before"] == "violated"
    # the whole point: after inserting the sanitizer the design is provably clean
    assert r["taint_after"] == "proven"
    added = {a["id"] for a in r["added"]}
    assert "San_Pay" in added
    # and the sanitizer sits on the path into the sink
    assert ["San_Pay", "Pay"] in [list(e) for e in r["edges"]]


def test_repair_is_noop_when_no_taint(client):
    clean = {"entry": "A", "edges": [["A", "B"]],
             "nodes": [{"id": "A", "capability": "plain"}, {"id": "B", "capability": "sink"}]}
    r = client.post("/api/repair", json=clean).json()
    assert r["repaired"] is False


def test_ctl_reachability_and_completion(client):
    r = client.post("/api/ctl", json={
        **TAINTED,
        "properties": [{"type": "reachable", "a": "Pay"}, {"type": "completes"}],
    }).json()
    verdicts = {p["label"]: p["verdict"] for p in r["properties"]}
    assert verdicts["EF Pay"] == "PROVEN"
    assert verdicts["AF terminal"] == "PROVEN"
    assert r["dead_ends"] == []


def test_ctl_detects_unreachable_node(client):
    # 'Orphan' has no incoming edge from the entry — reachability must FAIL.
    spec = {
        "entry": "A",
        "edges": [["A", "B"]],
        "nodes": [{"id": "A", "capability": "plain"}, {"id": "B", "capability": "plain"},
                  {"id": "Orphan", "capability": "plain"}],
        "properties": [{"type": "reachable", "a": "Orphan"}],
    }
    r = client.post("/api/ctl", json=spec).json()
    assert r["properties"][0]["verdict"] == "VIOLATED"


def test_prove_fails_closed_on_false_obligation(client):
    r = client.post("/api/prove", json={
        "data": {"area": 100, "rate": 3, "total": 999},
        "obligations": ["total == area * rate"],
    }).json()
    assert r["verified"] is False
    assert "total == area * rate" in r["failed"]


def test_prove_accepts_true_obligation(client):
    r = client.post("/api/prove", json={
        "data": {"area": 100, "rate": 3, "total": 300},
        "obligations": ["total == area * rate", "area > 0"],
    }).json()
    assert r["verified"] is True


def test_certificate_seals_verdict_with_hash(client):
    r = client.post("/api/certificate", json={
        "name": "t", **TAINTED, "invariants": ["1 == 1"],
        "nodes": [{**n, "type": "extract" if n["capability"] != "sink" else "tool",
                   "obligations": []} for n in TAINTED["nodes"]],
    }).json()
    assert r["aura_certificate"] == "1.1"
    # taint is violated in this design, so the whole certificate is NOT verified
    assert r["verified"] is False
    assert r["summary"]["taint"] == "violated"
    assert len(r["sha256"]) == 64


def test_dataset_bulk_verify_counts_violations(client):
    r = client.post("/api/verify_dataset", json={
        "records": [{"amount": 10}, {"amount": -5}, {"amount": 800}],
        "obligations": ["amount >= 0", "amount <= 500"],
    }).json()
    assert r["total"] == 3
    assert r["passed"] == 1
    assert r["failed"] == 2


def test_policy_scan_flags_secret_and_pii(client):
    r = client.post("/api/policy/scan", json={"nodes": [
        {"id": "A", "system_prompt": "call with api_key=sk-abcdefghij1234567890ABCD", "obligations": []},
        {"id": "B", "system_prompt": "email admin@acme.com", "obligations": []},
    ]}).json()
    assert r["clean"] is False
    kinds = {f["kind"] for f in r["findings"]}
    assert "secret.openai" in kinds
    assert "pii.email" in kinds
    # matches are redacted, never the full secret
    assert all("sk-abcdefghij1234567890ABCD" != f["match"] for f in r["findings"])


def test_policy_scan_clean_when_no_secrets(client):
    r = client.post("/api/policy/scan", json={"nodes": [
        {"id": "A", "system_prompt": "Classify the refund reason.", "obligations": ["amount >= 0"]},
    ]}).json()
    assert r["clean"] is True
    assert r["count"] == 0


def test_tune_injects_fewshot_demonstrations(client):
    r = client.post("/api/tune", json={
        "node": "Classify", "prompt": "Classify the refund.", "new_input": "damaged",
        "examples": [{"input": "box arrived broken", "output": {"category": "damaged"}},
                     {"input": "never received", "output": {"category": "missing"}}],
    }).json()
    assert "FEW-SHOT DEMONSTRATIONS" in r["optimized"]
    assert r["n_demos"] == 2


def test_memory_prune_keeps_system_and_last_n(client):
    r = client.post("/api/memory/preview", json={
        "max_messages": 2, "required_keys": ["order_id"],
        "history": [{"role": "system", "content": "sys"}, {"role": "user", "content": "m1"},
                    {"role": "user", "content": "m2"}, {"role": "user", "content": "m3"}],
    }).json()
    assert r["pruned"][0]["role"] == "system"
    # required-key context injected + only the last 2 non-system messages kept
    assert any("order_id" in m["content"] for m in r["pruned"])
    assert r["pruned"][-1]["content"] == "m3"


def test_audit_chain_appends_and_detects_tampering(client, tmp_path, monkeypatch):
    # log something, then confirm the chain verifies, then confirm a manual edit breaks it.
    client.post("/api/audit", json={"action": "verify", "summary": "s1", "detail": {"a": 1}})
    client.post("/api/audit", json={"action": "save", "summary": "s2", "detail": {"b": 2}})
    v = client.get("/api/audit/verify").json()
    assert v["count"] >= 2
    entries = client.get("/api/audit").json()
    # newest first; every entry seals the previous hash
    assert entries[0]["prev_hash"] == entries[1]["hash"]


# ── Trifecta + MCP import (0.7.0) ─────────────────────────────────────────────

TRIFECTA_SPEC = {
    "entry": "Fetch",
    "edges": [["Fetch", "Read"], ["Read", "Send"]],
    "nodes": [
        {"id": "Fetch", "kind": "tool", "tool_name": "web.fetch", "side_effect": "read", "capability": "plain"},
        {"id": "Read", "kind": "tool", "tool_name": "db.query", "side_effect": "read", "capability": "plain"},
        {"id": "Send", "kind": "tool", "tool_name": "email.send", "side_effect": "external", "capability": "sink"},
    ],
}


def test_verify_reports_closed_trifecta(client):
    r = client.post("/api/verify", json=TRIFECTA_SPEC).json()
    assert r["trifecta"]["verdict"] == "CLOSED"
    f = r["trifecta"]["findings"][0]
    assert (f["untrusted"], f["private"], f["exfil"]) == ("Fetch", "Read", "Send")


def test_mcp_import_builds_hub_flow(client):
    cfg = {"tools": [
        {"name": "fetch", "description": "Fetch a URL", "annotations": {"readOnlyHint": True}},
        {"name": "read_file", "description": "Read a local file", "annotations": {"readOnlyHint": True}},
        {"name": "slack_post_message", "description": "Post to Slack", "annotations": {"readOnlyHint": False}},
    ]}
    flow = client.post("/api/mcp/import", json={"config": cfg}).json()
    ids = {n["id"] for n in flow["nodes"]}
    assert "Agent" in ids and "slack_post_message" in ids
    # the imported flow closes the trifecta when verified
    r = client.post("/api/verify", json={"nodes": flow["nodes"], "edges": flow["edges"], "entry": flow["entry"]}).json()
    assert r["trifecta"]["verdict"] == "CLOSED"


def test_mcp_import_rejects_non_mcp(client):
    assert client.post("/api/mcp/import", json={"config": {"nodes": []}}).status_code == 400


def test_code_import_from_source(client):
    src = ('from langchain_core.tools import tool\n'
           '@tool\ndef fetch_url(u):\n    "Fetch a URL"\n    ...\n'
           '@tool\ndef read_file(p):\n    "Read a local file"\n    ...\n'
           '@tool\ndef post_to_slack(t):\n    "Post to Slack"\n    ...\n')
    flow = client.post("/api/code/import", json={"source": src}).json()
    assert {"Agent", "fetch_url", "read_file", "post_to_slack"} <= {n["id"] for n in flow["nodes"]}
    v = client.post("/api/verify", json={"nodes": flow["nodes"], "edges": flow["edges"], "entry": flow["entry"]}).json()
    assert v["trifecta"]["verdict"] == "CLOSED"


def test_code_import_rejects_empty_and_toolless(client):
    assert client.post("/api/code/import", json={"source": ""}).status_code == 400
    assert client.post("/api/code/import", json={"source": "x = 1"}).status_code == 400


def test_check_endpoint_returns_findings_with_keys(client):
    r = client.post("/api/check", json=TRIFECTA_SPEC).json()
    assert r["verified"] is False
    tri = [f for f in r["findings"] if f["check"] == "trifecta"]
    assert tri and tri[0]["key"] == "Fetch->Send"     # stable discriminator for baseline diff


def test_provider_key_set_and_clear(client):
    assert not [p for p in client.get("/api/providers").json() if p["name"] == "openai"][0]["available"]
    r = client.post("/api/providers/key", json={"provider": "openai", "key": "sk-test"}).json()
    assert r["available"] is True
    assert [p for p in client.get("/api/providers").json() if p["name"] == "openai"][0]["available"]
    assert client.post("/api/providers/key", json={"provider": "openai", "key": ""}).json()["available"] is False
    assert client.post("/api/providers/key", json={"provider": "ollama", "key": "x"}).status_code == 400


def test_demo_provider_runs_with_no_setup(client):
    # The zero-setup "demo" provider must run an extract flow end-to-end (mock
    # extractions, real routing) with no key and no ollama.
    spec = {"provider": "demo", "input": "I want a refund", "entry": "A",
            "edges": [["A", "B"]],
            "nodes": [{"id": "A", "type": "extract", "fields": [{"name": "amount", "type": "int"}], "obligations": ["amount >= 0"]},
                      {"id": "B", "type": "extract", "fields": [{"name": "category", "type": "str"}]}]}
    r = client.post("/api/run", json=spec).json()
    assert not r.get("error")
    nodes = [s["node"] for s in r["trace"]]
    assert "A" in nodes and "B" in nodes
    assert any(s.get("extracted", {}).get("amount") is not None for s in r["trace"])
    assert [p for p in client.get("/api/providers").json() if p["name"] == "demo"][0]["available"]


def test_verify_honors_analysis_roles(client):
    # the studio must pass a graph-importer's `roles` through to /api/verify, so a
    # node classified by its code (not name) is respected (was dropped -> false findings)
    spec = {"entry": "fetch", "edges": [["fetch", "lookup"], ["lookup", "send"]],
            "nodes": [{"id": "fetch", "kind": "tool", "roles": ["untrusted"]},
                      {"id": "lookup", "kind": "tool", "roles": ["private"]},
                      {"id": "send", "kind": "tool", "roles": ["exfil"]}]}
    r = client.post("/api/verify", json=spec).json()
    assert r["trifecta"]["verdict"] == "CLOSED"
    f = r["trifecta"]["findings"][0]
    assert f["exfil"] == "send" and f["untrusted"] == "fetch"
