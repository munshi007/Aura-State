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


def test_prose_system_prompt_does_not_trigger_roles():
    # a payment sink whose PROMPT reads "Issue the refund to the customer account"
    # must not gain untrusted/private roles from that prose — classification uses
    # tool_name + description only.
    roles, _ = classify_roles({"id": "Pay", "kind": "tool", "tool_name": "payment.refund",
                               "side_effect": "external",
                               "system_prompt": "Issue the refund to the customer account."})
    assert roles == {"exfil"}


# ── Pre-launch fail-open regressions (found by adversarial fuzz) ──────────────

def test_tool_name_classified_regardless_of_kind():
    # F1: a node that CALLS a tool must be classified even if kind != "tool"
    # (labelling send_email kind="decision" used to make it vanish -> fail-open).
    roles, _ = classify_roles({"id": "notify", "kind": "decision",
                               "tool_name": "send_email", "description": "send an email externally"})
    assert "exfil" in roles


def test_mislabeled_sanitizer_does_not_erase_sink():
    # F2: a node marked sanitizer that also looks like an exfil sink must NOT be
    # trusted as a pure sanitizer (it would prune the taint walk and hide the sink).
    roles, unk = classify_roles({"id": "S", "kind": "sanitizer",
                                 "tool_name": "send_email", "side_effect": "external"})
    assert "sanitizer" not in roles and "exfil" in roles and unk is True


def test_lethal_flow_with_decision_labeled_sink_is_flagged():
    from aura_state.check import check_flow
    flow = {"entry": "A", "edges": [["A", "F"], ["F", "R"], ["R", "N"]],
            "nodes": [{"id": "A", "kind": "extract", "capability": "plain"},
                      {"id": "F", "kind": "tool", "tool_name": "fetch_web", "description": "scrape a website"},
                      {"id": "R", "kind": "tool", "tool_name": "query_db", "description": "query the customer database"},
                      {"id": "N", "kind": "decision", "tool_name": "send_email", "description": "send an email externally"}]}
    assert check_flow(flow).verified is False


def test_unknown_tools_with_both_legs_cannot_be_proven_safe():
    # F3: unclassifiable tool + reachable untrusted + private -> fail closed
    from aura_state.check import check_flow
    flow = {"entry": "A", "edges": [["A", "F"], ["F", "R"], ["R", "X"]],
            "nodes": [{"id": "A", "kind": "extract", "capability": "plain"},
                      {"id": "F", "kind": "tool", "tool_name": "fetch_url", "side_effect": "read"},
                      {"id": "R", "kind": "tool", "tool_name": "read_customer_file", "side_effect": "read"},
                      {"id": "X", "kind": "tool", "tool_name": "zzz_custom_op"}]}   # unknown, no side_effect
    r = check_flow(flow)
    assert r.verified is False
    assert any(f.check == "trifecta" and f.severity == "high" for f in r.findings)


def test_import_with_no_tools_is_not_reported_green():
    from aura_state.check import check_flow
    r = check_flow({"source": "code", "entry": "Agent",
                    "nodes": [{"id": "Agent", "kind": "extract", "capability": "plain"}], "edges": []})
    assert any(f.check == "structure" for f in r.findings)


def test_name_only_fetch_stays_untrusted_no_body_to_disambiguate():
    # The name-only path (MCP/CrewAI tools) has no body, so a tool literally named
    # `fetch` MUST stay untrusted — it is the canonical web-fetch source in the
    # lethal-trifecta example. (The body-aware graph importer drops the bare token
    # precisely because it can see requests.get vs db.execute; this path cannot,
    # so it fails CLOSED. Regression guard for that deliberate asymmetry.)
    roles, _ = classify_roles({"kind": "tool", "tool_name": "fetch",
                               "description": "fetch a URL"})
    assert "untrusted" in roles
