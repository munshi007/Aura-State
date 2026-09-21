"""Graph-aware static import for LangGraph agents.

The plain tool importer sees only `@tool` functions and misses the agent's real
structure — the StateGraph nodes/edges and, crucially, actions written as graph
*node functions* (a `send_email(state)` node is an exfil leg the tool scan never
sees). This module parses the `StateGraph` with `ast` (never executing it) into a
faithful flow: the real nodes, the real edges (including conditional edges), and a
role for each node inferred from WHAT ITS CODE DOES — reads a file/DB (private),
fetches the web / an issue (untrusted), sends mail / posts / runs code (exfil) —
so the trifecta runs over the actual control flow instead of a worst-case hub.

Still 100% design-time and static: nothing is imported or run.
"""
from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Set

# What a node's code DOES — matched against its source text. Method-aware where it
# matters (`requests.get` ingests, `requests.post` exfiltrates).
_EXFIL = re.compile(
    r"(requests|httpx|session|client|urllib)\.(post|put|patch)\b"
    r"|\bsend\w*\(|sendmail|smtp|\bemail\b|send_email|\.publish\b|\.upload\b|webhook"
    r"|slack|discord|telegram|\bsms\b|twilio|sendgrid|\bses\b|\bsns\b|\bsqs\b|notify"
    r"|tweet|create_issue|add_\w*comment|create_pull|\.push\b", re.I)
_UNTRUSTED = re.compile(
    r"(requests|httpx|session|client|urllib|http)\.(get|head)\b|urlopen"
    r"|\bfetch\w*|scrape\w*|crawl\w*|browse\w*|download\w*|tavily|serper|brave"
    r"|bing|google\w*search|read_website|read_url|get_issue|list_issues|feedparser"
    r"|\bimap\b|\binbox\b|websearch|web_search", re.I)
_PRIVATE = re.compile(
    r"\bopen\s*\(|\.read\w*\(|\.load\w*\(|\.query\w*\(|\.execute\w*\(|fetchall|fetchone"
    r"|cursor|getenv|environ|secret|vault|boto3|\bs3\b|get_object|read_file"
    r"|get_file_contents|postgres|mysql|mongo\b|redis|retriev\w*|vectorstore|embed\w*"
    r"|customer|account|knowledge|\bkb\b", re.I)

_GRAPH_CTORS = {"StateGraph", "Graph", "MessageGraph"}


def _callee(call: ast.Call) -> Optional[str]:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _end(node: ast.AST) -> Optional[str]:
    """Resolve an edge endpoint arg to a node name / START / END."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return node.id            # START / END / an aliased constant
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _role_signals(text: str) -> Set[str]:
    roles: Set[str] = set()
    if _EXFIL.search(text):
        roles.add("exfil")
    if _UNTRUSTED.search(text):
        roles.add("untrusted")
    if _PRIVATE.search(text):
        roles.add("private")
    return roles


def _fn_text(fn: ast.AST) -> str:
    doc = ast.get_docstring(fn) or ""
    try:
        return doc + "\n" + ast.unparse(fn)
    except Exception:
        return doc


def langgraph_flow(source: str, name: str = "code-agent") -> Optional[Dict[str, Any]]:
    """Return a faithful flow for a LangGraph StateGraph agent, or None if the
    source doesn't build one (caller falls back to the tool-surface importer)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    funcs = {n.name: n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    graph_vars: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if _callee(node.value) in _GRAPH_CTORS:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        graph_vars.add(t.id)
    if not graph_vars:
        return None

    def is_method(call: ast.Call, meth: str) -> bool:
        f = call.func
        return (isinstance(f, ast.Attribute) and f.attr == meth
                and isinstance(f.value, ast.Name) and f.value.id in graph_vars)

    node_fn: Dict[str, Optional[str]] = {}   # node name -> function name
    edges: List[List[str]] = []
    entry: Optional[str] = None
    conditional_srcs: List[str] = []         # sources whose targets we couldn't resolve

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if is_method(node, "add_node") and node.args:
            nm = None
            fn = None
            if isinstance(node.args[0], ast.Constant):
                nm = node.args[0].value
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Name):
                    fn = node.args[1].id
            elif isinstance(node.args[0], ast.Name):     # add_node(func) -> name = func name
                nm = fn = node.args[0].id
            if nm:
                node_fn[nm] = fn
        elif is_method(node, "add_edge") and len(node.args) >= 2:
            a, b = _end(node.args[0]), _end(node.args[1])
            if a and b:
                if a == "START":
                    entry = b
                elif b != "END":
                    edges.append([a, b])
        elif is_method(node, "add_conditional_edges") and node.args:
            src = _end(node.args[0])
            mapping = node.args[2] if len(node.args) >= 3 else None
            for k in node.keywords:
                if k.arg in ("path_map", "conditional_edge_mapping"):
                    mapping = k.value
            if isinstance(mapping, ast.Dict):
                for v in mapping.values:
                    tgt = _end(v)
                    if tgt and tgt not in ("END",):
                        edges.append([src, tgt])
            elif src:
                conditional_srcs.append(src)   # router targets unknown -> resolve below
        elif is_method(node, "set_entry_point") and node.args:
            entry = _end(node.args[0])
        elif is_method(node, "set_finish_point"):
            pass

    if not node_fn:
        return None

    # A conditional edge with a dynamic router (no static mapping) could go to any
    # node — connect it to all of them, so the analysis never MISSES a real path
    # (sound over-approximation, the security-safe choice).
    all_names = list(node_fn.keys())
    for src in conditional_srcs:
        for tgt in all_names:
            if tgt != src and [src, tgt] not in edges:
                edges.append([src, tgt])

    flow_nodes: List[Dict[str, Any]] = []
    for nm, fn in node_fn.items():
        text = nm + " " + (fn or "")
        doc = ""
        if fn and fn in funcs:
            doc = ast.get_docstring(funcs[fn]) or ""
            text += " " + _fn_text(funcs[fn])
        roles = _role_signals(text)
        obj: Dict[str, Any] = {"id": nm, "kind": "tool" if roles else "extract",
                               "description": doc or nm}
        if roles:
            obj["roles"] = sorted(roles)
        flow_nodes.append(obj)

    entry = entry or (flow_nodes[0]["id"] if flow_nodes else None)
    return {"name": name, "entry": entry, "nodes": flow_nodes,
            "edges": [e for e in edges if e[0] and e[1]], "source": "code"}
