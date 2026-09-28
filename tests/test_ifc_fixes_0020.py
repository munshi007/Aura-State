"""0020: taint → IFC integrity check unified onto the trifecta's role model.

Closes the residual fail-open: an untrusted tool classified by NAME / `data_class`
/ `roles` (not an explicit `capability`) is now a taint source, and the studio
`/api/verify` and the CLI `check` decide it the same way.
"""
import pytest

from aura_state.check import check_flow
from aura_state.verification.ifc import analyze_ifc


# untrusted by NAME only (no capability/data_class/roles) reaching a sink.
_NAME_ONLY = {
    "name": "t", "entry": "U", "edges": [["U", "X"]],
    "nodes": [{"id": "U", "kind": "tool", "tool_name": "web_fetch"},
              {"id": "X", "kind": "tool", "tool_name": "file_write", "side_effect": "write"}],
}


def test_name_classified_untrusted_is_a_taint_source_fixes_0020():
    # The capability-only taint missed this (no explicit untrusted marking); the
    # role-based IFC catches it, consistent with the trifecta classifier.
    r = check_flow(_NAME_ONLY)
    assert r.verified is False
    assert any(f.check == "taint" and f.node == "X" for f in r.findings)


def test_ifc_integrity_flow_and_sanitizer_clears_it_fixes_0020():
    res = analyze_ifc(_NAME_ONLY["nodes"], _NAME_ONLY["edges"], "U")
    assert res.verified is False
    assert res.flows and res.flows[0].source == "U" and res.flows[0].sink == "X"
    assert res.flows[0].external is False   # file_write is a local-write sink
    # add a sanitizer between -> cleared
    clean = {"nodes": [{"id": "U", "kind": "tool", "tool_name": "web_fetch"},
                       {"id": "S", "kind": "sanitizer"},
                       {"id": "X", "kind": "tool", "tool_name": "file_write", "side_effect": "write"}],
             "edges": [["U", "S"], ["S", "X"]]}
    assert analyze_ifc(clean["nodes"], clean["edges"], "U").verified is True


def test_cli_check_and_studio_verify_agree_on_injection_fixes_0020():
    # Same graph, two surfaces: both must flag it (was the CLI/studio disagreement).
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from aura_state.ui.server import create_app
    client = TestClient(create_app())

    cli_bad = not check_flow(_NAME_ONLY).verified
    v = client.post("/api/verify", json=_NAME_ONLY).json()
    studio_bad = v["taint"]["verdict"] == "VIOLATED"
    assert cli_bad and studio_bad
    assert any(fl["source"] == "U" and fl["sink"] == "X" for fl in v["taint"]["violations"])


def test_certificate_and_repair_use_the_unified_ifc_fixes_0020():
    # The studio /api/certificate and /api/repair must agree with verify/check on
    # the residual-fail-open case (name-only untrusted -> sink) — no surface left
    # on the pre-IFC capability taint.
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from aura_state.ui.server import create_app
    client = TestClient(create_app())

    cert = client.post("/api/certificate", json={
        "name": "t", "nodes": _NAME_ONLY["nodes"], "edges": _NAME_ONLY["edges"],
        "entry": "U", "invariants": []}).json()
    assert cert["taint"]["verdict"] == "violated" and cert["verified"] is False

    rep = client.post("/api/repair", json=_NAME_ONLY).json()
    assert rep["repaired"] is True and rep["taint_after"] == "proven"   # sanitizer inserted, re-proven
