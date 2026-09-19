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
    assert _side_effect({"readOnlyHint": True, "destructiveHint": True}) == "external"
    assert _side_effect({}) == "external"


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
