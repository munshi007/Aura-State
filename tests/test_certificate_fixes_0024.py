"""0024: verifiable proof certificate — trustless re-verification + tamper detection."""
import copy

from aura_state.certificate import make_certificate, verify_certificate


_VULN = {"name": "v", "entry": "U", "edges": [["U", "X"]],
         "nodes": [{"id": "U", "kind": "tool", "tool_name": "web.fetch", "side_effect": "read", "capability": "untrusted"},
                   {"id": "X", "kind": "tool", "tool_name": "email.send", "side_effect": "external", "capability": "sink"}]}
_CLEAN = {"name": "c", "entry": "U", "edges": [["U", "S"], ["S", "X"]],
          "nodes": [{"id": "U", "kind": "tool", "tool_name": "web.fetch", "side_effect": "read", "capability": "untrusted"},
                    {"id": "S", "kind": "sanitizer"},
                    {"id": "X", "kind": "tool", "tool_name": "email.send", "side_effect": "external"}]}


def _cert(flow):
    return make_certificate(flow, version="test", timestamp="2026-09-28T00:00:00")


def test_roundtrip_valid_fixes_0024():
    assert verify_certificate(_cert(_VULN))["valid"] is True
    assert verify_certificate(_cert(_CLEAN))["valid"] is True


def test_proven_agent_certifies_verified_fixes_0024():
    c = _cert(_CLEAN)
    assert c["verdict"]["verified"] is True
    v = _cert(_VULN)
    assert v["verdict"]["verified"] is False


def test_tampered_verdict_is_caught_fixes_0024():
    c = _cert(_VULN)                     # honestly "not proven"
    c["verdict"]["verified"] = True      # forge a "proven" claim
    res = verify_certificate(c)
    assert res["valid"] is False
    assert any("re-verification disagrees" in p for p in res["problems"])


def test_tampered_design_is_caught_fixes_0024():
    c = _cert(_VULN)
    c["design"]["nodes"].append({"id": "Z", "kind": "tool", "tool_name": "noop", "side_effect": "read"})
    res = verify_certificate(c)
    assert res["valid"] is False
    assert any("design hash mismatch" in p for p in res["problems"])


def test_swapping_a_clean_design_under_a_proven_claim_is_caught_fixes_0024():
    # forge: take a "proven" verdict but swap in the vulnerable design
    good = _cert(_CLEAN)
    good["design"] = copy.deepcopy(_VULN_DESIGN := {"name": "v", "entry": "U",
        "edges": [["U", "X"]], "nodes": _VULN["nodes"], "invariants": [], "manifest": None})
    res = verify_certificate(good)
    assert res["valid"] is False   # hash mismatch and/or re-verification catches it


def test_not_a_certificate_fixes_0024():
    assert verify_certificate({"hello": "world"})["valid"] is False
    assert verify_certificate({"aura_certificate": "1.0"})["valid"] is False
