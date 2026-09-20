"""Tests for the MCP tool-surface importer."""
from aura_state.loaders.mcp import flow_from_mcp, is_mcp_config, _side_effect
from aura_state.check import check_flow


TOOLS = {"tools": [
    {"name": "fetch", "description": "Fetch a URL", "annotations": {"readOnlyHint": True, "openWorldHint": True}},
    {"name": "read_file", "description": "Read a local file", "annotations": {"readOnlyHint": True}},
    {"name": "slack_post_message", "description": "Post to Slack", "annotations": {"readOnlyHint": False}},
]}


def test_detects_mcp_shapes_and_not_flows():
    assert is_mcp_config(TOOLS) is True
    assert is_mcp_config({"mcpServers": {"s": {"tools": []}}}) is True
    assert is_mcp_config([{"name": "x"}]) is True
    assert is_mcp_config({"nodes": [], "edges": []}) is False   # an Aura flow


def test_annotations_map_to_side_effect():
    assert _side_effect({"readOnlyHint": True}) == "read"
    assert _side_effect({"readOnlyHint": True, "destructiveHint": True}) == "read"     # read-only wins
    assert _side_effect({"openWorldHint": True}) == "external"                          # acts externally
    assert _side_effect({"openWorldHint": False}) == "write"                            # explicitly local
    assert _side_effect({}) is None                                                     # unknown reach -> fail-closed


def test_unannotated_lethal_surface_is_not_silently_safe():
    # regression: fetch + read_file + create_issue with NO annotations must not
    # verify as safe (the 0.7.1 fail-open: _side_effect defaulted to 'write').
    cfg = {"tools": [
        {"name": "fetch", "description": "Fetch a URL"},
        {"name": "read_file", "description": "Read a local file"},
        {"name": "create_issue", "description": "Create a GitHub issue"},
    ]}
    r = check_flow(flow_from_mcp(cfg))
    assert r.verified is False
    assert [f for f in r.findings if f.check == "trifecta" and f.severity == "critical"]


def test_collision_across_servers_keeps_both_tools():
    cfg = {"mcpServers": {
        "web": {"tools": [{"name": "search", "description": "Search the web", "annotations": {"openWorldHint": True}}]},
        "docs": {"tools": [{"name": "search", "description": "Search internal customer records", "annotations": {"readOnlyHint": True}}]},
    }}
    flow = flow_from_mcp(cfg)
    tool_ids = [n["id"] for n in flow["nodes"] if n["kind"] == "tool"]
    assert len(tool_ids) == 2 and len(set(tool_ids)) == 2   # neither dropped


def test_readonly_name_match_is_not_exfil():
    # slack_list_channels is read-only — "slack" in the name must NOT make it an exfil sink
    from aura_state.verification.trifecta import classify_roles
    roles, _ = classify_roles({"id": "slack_list_channels", "kind": "tool",
                               "tool_name": "slack_list_channels", "side_effect": "read"})
    assert "exfil" not in roles


def test_local_write_is_not_exfil():
    from aura_state.verification.trifecta import classify_roles
    roles, _ = classify_roles({"id": "write_file", "kind": "tool",
                               "tool_name": "write_file", "side_effect": "write"})
    assert "exfil" not in roles


def test_hub_graph_makes_every_tool_reachable_both_ways():
    flow = flow_from_mcp(TOOLS)
    ids = {n["id"] for n in flow["nodes"]}
    assert "Agent" in ids and {"fetch", "read_file", "slack_post_message"} <= ids
    # planner reaches every tool and every tool returns to the planner
    for t in ("fetch", "read_file", "slack_post_message"):
        assert ["Agent", t] in flow["edges"] and [t, "Agent"] in flow["edges"]


def test_fetch_plus_files_plus_slack_closes_the_trifecta():
    r = check_flow(flow_from_mcp(TOOLS, name="dev-assistant"))
    assert r.verified is False
    tri = [f for f in r.findings if f.check == "trifecta" and f.severity == "critical"]
    assert tri and tri[0].node == "slack_post_message"


def test_read_only_tool_surface_is_not_a_trifecta():
    # fetch + read_file, but nothing can communicate externally
    safe = {"tools": [
        {"name": "fetch", "description": "Fetch a URL", "annotations": {"readOnlyHint": True}},
        {"name": "read_file", "description": "Read a local file", "annotations": {"readOnlyHint": True}},
    ]}
    r = check_flow(flow_from_mcp(safe))
    assert not [f for f in r.findings if f.check == "trifecta" and f.severity == "critical"]


def test_malformed_tool_inputs_do_not_crash():
    # non-dict annotations, unhashable/missing names -> skipped cleanly, no crash
    flow = flow_from_mcp({"tools": [
        {"name": "t", "annotations": ["readOnlyHint"]},
        {"name": {"x": 1}},            # unhashable name -> skipped
        {"description": "no name"},    # missing name -> skipped
        {"name": "ok", "annotations": {"readOnlyHint": True}},
    ]})
    ids = {n["id"] for n in flow["nodes"]}
    assert "ok" in ids and "t" in ids
