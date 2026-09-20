"""Tests for the lethal-trifecta analyzer.

Each builds an adversarial agent and asserts whether the trifecta *closes*
(CLAUDE.md rule 8: exercise the real analysis against an input that should fail).
"""
from aura_state.verification.trifecta import analyze_trifecta, classify_roles
from aura_state.check import check_flow


def _tool(id, name, se):
    return {"id": id, "kind": "tool", "tool_name": name, "side_effect": se}


# web page (untrusted) -> read customer DB (private) -> email out (exfil)
TRIFECTA = {
    "name": "assistant", "entry": "Fetch",
    "edges": [["Fetch", "Read"], ["Read", "Send"]],
    "nodes": [
        _tool("Fetch", "web.fetch", "read"),
        _tool("Read", "db.query", "read"),
        _tool("Send", "email.send", "external"),
    ],
}


def test_classic_trifecta_closes():
    r = analyze_trifecta(TRIFECTA["nodes"], TRIFECTA["edges"], "Fetch")
    assert r.verified is False
    f = r.findings[0]
    assert (f.untrusted, f.private, f.exfil) == ("Fetch", "Read", "Send")
    assert f.shares_path is True
    assert f.path == ["Fetch", "Read", "Send"]


def test_sanitizer_breaks_the_trifecta():
    flow = {
        "name": "guarded", "entry": "Fetch",
        "edges": [["Fetch", "Guard"], ["Guard", "Read"], ["Read", "Send"]],
        "nodes": [
            _tool("Fetch", "web.fetch", "read"),
            {"id": "Guard", "kind": "sanitizer"},
            _tool("Read", "db.query", "read"),
            _tool("Send", "email.send", "external"),
        ],
    }
    r = analyze_trifecta(flow["nodes"], flow["edges"], "Fetch")
    assert r.verified is True   # untrusted taint is cleaned before the sink


def test_no_private_data_is_not_a_full_trifecta():
    # untrusted -> exfil (an injection path) but nothing private in scope
    flow = {
        "name": "echo", "entry": "Fetch",
        "edges": [["Fetch", "Send"]],
        "nodes": [_tool("Fetch", "web.fetch", "read"), _tool("Send", "http.post", "external")],
    }
    r = analyze_trifecta(flow["nodes"], flow["edges"], "Fetch")
    assert r.verified is True


def test_data_class_override_wins_over_name():
    # a blandly named tool the heuristic can't place, tagged private explicitly
    flow = {
        "name": "override", "entry": "Fetch",
        "edges": [["Fetch", "Load"], ["Load", "Send"]],
        "nodes": [
            _tool("Fetch", "web.fetch", "read"),
            {"id": "Load", "kind": "tool", "tool_name": "acme.lookup", "side_effect": "read",
             "data_class": "private"},
            _tool("Send", "email.send", "external"),
        ],
    }
    r = analyze_trifecta(flow["nodes"], flow["edges"], "Fetch")
    assert r.verified is False
    assert r.findings[0].private == "Load"


def test_unclassified_read_tool_is_surfaced():
    roles, unk = classify_roles({"id": "X", "kind": "tool", "tool_name": "acme.doThing", "side_effect": "read"})
    assert unk is True and not roles


def test_classify_roles_basic():
    assert "untrusted" in classify_roles(_tool("a", "web.fetch", "read"))[0]
    assert "private" in classify_roles(_tool("b", "db.query", "read"))[0]
    assert "exfil" in classify_roles(_tool("c", "email.send", "external"))[0]
    assert "exfil" in classify_roles(_tool("d", "s3.put", "write"))[0]


def test_check_flow_reports_trifecta_as_critical():
    r = check_flow(TRIFECTA)
    assert r.verified is False
    tri = [f for f in r.findings if f.check == "trifecta" and f.severity == "critical"]
    assert tri and tri[0].node == "Send"
    assert r.summary.get("trifecta") == "closed"


# ── 0.7.2 regression: false-negatives the review found ────────────────────────

def _t(id, name, se):
    return {"id": id, "kind": "tool", "tool_name": name, "side_effect": se}


def test_compound_exfil_name_is_caught():
    # 'post_update' must register as exfil ('post\b' used to miss compound names)
    flow = {"nodes": [_t("scrape_web", "scrape_web", "read"),
                      _t("customer_lookup", "customer_lookup", "read"),
                      _t("post_update", "post_update", "write")],
            "edges": [["scrape_web", "customer_lookup"], ["customer_lookup", "post_update"]]}
    assert analyze_trifecta(flow["nodes"], flow["edges"], "scrape_web").verified is False


def test_read_and_send_single_node_closes_trifecta():
    # one tool that reads private data AND sends externally is a full trifecta;
    # private/untrusted must be classified even when side_effect == 'external'
    flow = {"nodes": [_t("read_inbox", "read_inbox", "read"),
                      _t("email_customer_record", "email_customer_record", "external")],
            "edges": [["read_inbox", "email_customer_record"]]}
    r = analyze_trifecta(flow["nodes"], flow["edges"], "read_inbox")
    assert r.verified is False


def test_postgres_is_not_exfil_via_post():
    roles, _ = classify_roles(_t("q", "postgres_query", "read"))
    assert "exfil" not in roles and "private" in roles


def test_local_write_stays_non_private_non_exfil():
    roles, _ = classify_roles(_t("w", "write_file", "write"))
    assert roles == set()
