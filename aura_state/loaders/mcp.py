"""Import an MCP tool surface into an Aura flow — audit what your agent *can do*.

Model Context Protocol agents hand the LLM a bag of tools (from one or more MCP
servers) and let it call them in any order it likes. So the honest, sound model
of "what can this agent do" is a **hub**: a planner node wired to every tool,
and every tool's result wired back to the planner — because any tool's output
can influence the next tool call. Over that graph, Aura's trifecta/taint asks
the question nobody else answers statically:

    You connected `fetch` + `filesystem` + `slack`. Together they close the
    lethal trifecta — a malicious web page can read your files and post them out.

This importer does NOT connect to or run any server (that would be execution).
It reads a *declaration* of the tools:

  * a `tools/list` result           →  {"tools": [{"name","description",...}]}
  * an MCP client config with tools →  {"mcpServers": {"srv": {"tools":[...]}}}
  * a bare list                     →  [{"name","description",...}, ...]

Each tool's read/write nature is taken from MCP annotations when present
(`readOnlyHint`, `destructiveHint`, `openWorldHint`); otherwise the name/desc
heuristic in :mod:`aura_state.verification.trifecta` classifies it.
"""
from __future__ import annotations

from typing import Any, Dict, List


AGENT = "Agent"   # the LLM planner hub


def _tools_from(config: Any) -> List[Dict[str, Any]]:
    """Pull a flat list of {server, name, description, annotations} tool defs
    out of any of the supported shapes."""
    out: List[Dict[str, Any]] = []

    def _add(t: Dict[str, Any], server: str = "") -> None:
        if not isinstance(t, dict) or "name" not in t:
            return
        out.append({
            "server": server,
            "name": t["name"],
            "description": t.get("description", ""),
            "annotations": t.get("annotations", {}) or {},
        })

    if isinstance(config, list):
        for t in config:
            _add(t)
    elif isinstance(config, dict):
        if isinstance(config.get("tools"), list):
            for t in config["tools"]:
                _add(t)
        if isinstance(config.get("mcpServers"), dict):
            for server, spec in config["mcpServers"].items():
                for t in (spec or {}).get("tools", []) or []:
                    _add(t, server)
    return out


def _side_effect(ann: Dict[str, Any]) -> str:
    """Map MCP tool annotations to an Aura side-effect.

    readOnlyHint=True   → 'read'      (never mutates; e.g. fetch, list, get_*)
    openWorldHint=True  → 'external'  (acts on outside systems → an exfil channel:
                           slack_post, add_issue_comment, http POST)
    openWorldHint=False → 'write'     (a local mutation like write_file — a
                           dangerous sink, but NOT the trifecta's exfil leg)
    no hints            → None        (UNKNOWN reach — do NOT assume local/safe;
                           None lets the classifier's name heuristics run and, if
                           still unplaceable, flags the tool as an advisory rather
                           than silently proving a lethal surface safe.)

    Erring toward 'write' on unannotated tools was a fail-open: it skipped the
    untrusted/private legs, so an unannotated fetch+read+send surface verified as
    safe. None is the fail-closed default.
    """
    if ann.get("readOnlyHint") is True:
        return "read"
    if ann.get("openWorldHint") is True:
        return "external"
    if ann.get("openWorldHint") is False:
        return "write"
    return None


def is_mcp_config(obj: Any) -> bool:
    """True if `obj` looks like an MCP tool declaration rather than an Aura flow.

    An Aura flow always has a `nodes` list; an MCP declaration has `tools` or
    `mcpServers`, or is a bare list of tool defs.
    """
    if isinstance(obj, list):
        return all(isinstance(t, dict) and "name" in t for t in obj)  # [] = empty surface
    if isinstance(obj, dict):
        if "nodes" in obj:
            return False
        return isinstance(obj.get("tools"), list) or isinstance(obj.get("mcpServers"), dict)
    return False


def flow_from_mcp(config: Any, name: str = "mcp-agent") -> Dict[str, Any]:
    """Build the worst-case Aura flow for an MCP tool surface (see module doc).

    The planner hub reaches every tool and every tool returns to the hub, so the
    analysis considers any order the LLM might call them in.
    """
    tools = _tools_from(config)
    nodes: List[Dict[str, Any]] = [
        {"id": AGENT, "kind": "extract", "capability": "plain",
         "description": "LLM planner — may call any tool in any order"}
    ]
    edges: List[List[str]] = []
    used: set = {AGENT}
    for t in tools:
        # Keep the bare tool name as the id (readable), but disambiguate real
        # collisions across servers instead of silently dropping the second tool
        # (dropping one hides a whole leg of the trifecta).
        nid = t["name"]
        if nid in used:
            base = f"{t['server']}:{t['name']}" if t["server"] else t["name"]
            nid, k = base, 2
            while nid in used:
                nid = f"{base}_{k}"; k += 1
        used.add(nid)
        nodes.append({
            "id": nid,
            "kind": "tool",
            "tool_name": t["name"],
            "side_effect": _side_effect(t["annotations"]),
            "description": (f"[{t['server']}] " if t["server"] else "") + t["description"],
        })
        edges.append([AGENT, nid])   # planner can invoke it
        edges.append([nid, AGENT])   # its result feeds the next decision
    return {"name": name, "entry": AGENT, "nodes": nodes, "edges": edges, "source": "mcp"}
