"""Import a real agent's tool surface from its SOURCE CODE — statically.

Point Aura at a LangGraph / CrewAI / LangChain agent (a file or a directory) and
it extracts the tools the agent can call, then builds the same worst-case hub
flow the MCP importer uses (the LLM planner may call any tool in any order) so
the lethal-trifecta / taint checks run against your real agent.

This parses the code with `ast` and NEVER imports or executes it — reading an
untrusted repo must not run it. What it recognises:

* `@tool` / `@tool("name")` decorated functions (LangChain/CrewAI/LlamaIndex share
  this) — name + docstring.
* `Tool(...)` / `StructuredTool.from_function(...)` — name + description kwargs.
* Framework tool classes instantiated by name (`SerperDevTool()`,
  `FileReadTool()`, `CodeInterpreterTool()`, …) — mapped to known roles.

Tools it can't confidently classify become advisories downstream (fail-closed),
never silent passes. Coverage is best-effort by design: a tool only referenced
dynamically won't be seen, so the report says how many tools were found.
"""
from __future__ import annotations

import ast
import os
from typing import Any, Dict, List, Optional

from .mcp import flow_from_mcp

# Known framework tool classes -> explicit role hints (name heuristics can't tell
# a local write from an external send, so we pin the well-known ones). Values map
# to node overrides consumed by trifecta.classify_roles.
_KNOWN_TOOLS: Dict[str, Dict[str, Any]] = {
    # web / untrusted content
    "SerperDevTool": {"desc": "search the web", "data_class": "untrusted"},
    "SerpApiTool": {"desc": "search the web", "data_class": "untrusted"},
    "EXASearchTool": {"desc": "search the web", "data_class": "untrusted"},
    "WebsiteSearchTool": {"desc": "search a website", "data_class": "untrusted"},
    "ScrapeWebsiteTool": {"desc": "scrape a website", "data_class": "untrusted"},
    "ScrapeElementFromWebsiteTool": {"desc": "scrape a website element", "data_class": "untrusted"},
    "BrowserbaseLoadTool": {"desc": "load a web page", "data_class": "untrusted"},
    "TavilySearchResults": {"desc": "search the web", "data_class": "untrusted"},
    # private data reads
    "FileReadTool": {"desc": "read a local file", "data_class": "private"},
    "DirectoryReadTool": {"desc": "read a local directory", "data_class": "private"},
    "PDFSearchTool": {"desc": "search a local PDF", "data_class": "private"},
    "DOCXSearchTool": {"desc": "search a local DOCX", "data_class": "private"},
    "CSVSearchTool": {"desc": "search a local CSV", "data_class": "private"},
    "PGSearchTool": {"desc": "query a Postgres database", "data_class": "private"},
    "MySQLSearchTool": {"desc": "query a MySQL database", "data_class": "private"},
    # local mutation (dangerous sink, but NOT the trifecta's external-comms leg)
    "FileWriterTool": {"desc": "write a local file", "side_effect": "write"},
    "FileWriteTool": {"desc": "write a local file", "side_effect": "write"},
    # external comms / code execution (exfil channel)
    "CodeInterpreterTool": {"desc": "execute arbitrary Python", "exfil": True},
    "ComposioTool": {"desc": "call an external SaaS action", "exfil": True},
    "EmailTool": {"desc": "send an email", "exfil": True},
}


def _is_tool_decorator(dec: ast.AST) -> bool:
    """True for @tool, @tool(...), @something.tool, @something.tool(...)."""
    target = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(target, ast.Name):
        return target.id == "tool"
    if isinstance(target, ast.Attribute):
        return target.attr == "tool"
    return False


def _str_arg(call: ast.Call, kw: str) -> Optional[str]:
    for k in call.keywords:
        if k.arg == kw and isinstance(k.value, ast.Constant) and isinstance(k.value.value, str):
            return k.value.value
    for a in call.args:                       # first positional string
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            return a.value
    return None


def _callee_name(call: ast.Call) -> Optional[str]:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _extract_tools(source: str) -> List[Dict[str, Any]]:
    """Return a list of tool dicts {name, description, [data_class|side_effect|exfil]}."""
    tree = ast.parse(source)
    found: Dict[str, Dict[str, Any]] = {}

    def add(name: str, desc: str = "", **role: Any) -> None:
        if not name or name in found:
            if name in found and desc and not found[name].get("description"):
                found[name]["description"] = desc
            return
        found[name] = {"name": name, "description": desc, **role}

    for node in ast.walk(tree):
        # @tool-decorated functions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if _is_tool_decorator(dec):
                    name = (_str_arg(dec, "name") if isinstance(dec, ast.Call) else None) or node.name
                    add(name, ast.get_docstring(node) or "")
                    break
        # Tool(...) / StructuredTool.from_function(...) and framework tool classes
        if isinstance(node, ast.Call):
            callee = _callee_name(node)
            if not callee:
                continue
            if callee in ("Tool", "StructuredTool") or callee.endswith("from_function"):
                add(_str_arg(node, "name") or "tool", _str_arg(node, "description") or "")
            elif callee in _KNOWN_TOOLS:
                spec = _KNOWN_TOOLS[callee]
                add(callee, spec.get("desc", ""),
                    **{k: v for k, v in spec.items() if k in ("data_class", "side_effect", "exfil")})
            elif callee[:1].isupper() and callee.endswith("Tool"):
                add(callee, "")   # unknown framework tool -> heuristic + advisory
    return list(found.values())


def _tool_name_of(el: ast.AST) -> Optional[str]:
    """Tool name from a `tools=[...]` element: a class instantiation or a @tool ref."""
    if isinstance(el, ast.Call):
        return _callee_name(el)
    if isinstance(el, ast.Name):
        return el.id
    if isinstance(el, ast.Attribute):
        return el.attr
    return None


def _crewai_flow(source: str, name: str) -> Optional[Dict[str, Any]]:
    """Model a MULTI-AGENT crew (CrewAI / AutoGen) with each agent's tools SCOPED
    to that agent, connected by the crew's hand-off order — instead of flattening
    every tool into one hub (which invents cross-agent paths that can't happen)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    agents: Dict[str, Dict[str, Any]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and _callee_name(node.value) in ("Agent", "AssistantAgent", "ConversableAgent"):
            var = node.targets[0].id if node.targets and isinstance(node.targets[0], ast.Name) else None
            if not var:
                continue
            role = _str_arg(node.value, "role") or _str_arg(node.value, "name") or var
            tools: List[str] = []
            for kw in node.value.keywords:
                if kw.arg in ("tools",) and isinstance(kw.value, ast.List):
                    tools = [t for t in (_tool_name_of(e) for e in kw.value.elts) if t]
            agents[var] = {"role": role, "tools": tools}
    if len(agents) < 2:
        return None   # a single agent is fine as a plain tool hub

    order = list(agents.keys())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _callee_name(node) == "Crew":
            for kw in node.keywords:
                if kw.arg == "agents" and isinstance(kw.value, ast.List):
                    order = [e.id for e in kw.value.elts if isinstance(e, ast.Name) and e.id in agents] or order

    nodes: List[Dict[str, Any]] = [{"id": "END", "kind": "extract"}][:0]   # typed empty
    edges: List[List[str]] = []
    prev_agent: Optional[str] = None
    for var in order:
        a = agents[var]
        aid = a["role"]
        nodes.append({"id": aid, "kind": "extract", "capability": "plain",
                      "description": f"agent: {a['role']}"})
        for t in a["tools"]:
            tid = f"{aid}:{t}"                     # scope the tool to THIS agent
            spec = _KNOWN_TOOLS.get(t, {})
            tnode: Dict[str, Any] = {"id": tid, "kind": "tool", "tool_name": t,
                                     "description": spec.get("desc", "")}
            for k in ("data_class", "side_effect", "exfil"):
                if k in spec:
                    tnode[k] = spec[k]
            nodes.append(tnode)
            edges.append([aid, tid]); edges.append([tid, aid])
        if prev_agent:                             # sequential hand-off between agents
            edges.append([prev_agent, aid])
        prev_agent = aid
    return {"name": name, "entry": nodes[0]["id"] if nodes else None,
            "nodes": nodes, "edges": edges, "source": "code"}


def _agent_tools_flow(source: str, name: str) -> Optional[Dict[str, Any]]:
    """AutoGen / AgentChat: `AssistantAgent(name=..., tools=[fn, fn2])` where the
    tools are plain FUNCTION REFERENCES (not @tool-decorated, not Tool() classes).

    The tool-hub importer only sees @tool/Tool()/`*Tool` classes, so a real
    single-agent AutoGen app imported as "no tools found". Here we resolve each
    referenced function to its body and classify it by WHAT THE CODE DOES — the
    same body-aware engine the LangGraph importer uses (requests.get → untrusted,
    db/read → private, send/post → exfil) — instead of dropping it. Static only.
    """
    from .graph_extract import _role_signals, _fn_text
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    funcs = {n.name: n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    ctors = ("AssistantAgent", "ConversableAgent", "Agent")
    tool_refs: List[str] = []
    seen: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _callee_name(node) in ctors:
            for kw in node.keywords:
                if kw.arg == "tools" and isinstance(kw.value, ast.List):
                    for e in kw.value.elts:
                        nm = _tool_name_of(e)
                        if nm and nm not in seen:
                            seen.add(nm)
                            tool_refs.append(nm)
    # Only claim this shape if at least one tool is a resolvable local function —
    # otherwise fall through to the tool-hub importer (which handles Tool() etc.).
    if not any(t in funcs for t in tool_refs):
        return None

    AGENT = "Agent"
    nodes: List[Dict[str, Any]] = [
        {"id": AGENT, "kind": "extract", "capability": "plain",
         "description": "LLM assistant — may call any tool in any order"}]
    edges: List[List[str]] = []
    for t in tool_refs:
        fn = funcs.get(t)
        doc = ast.get_docstring(fn) if fn else ""
        roles = _role_signals(t + " " + (_fn_text(fn) if fn else ""))
        tnode: Dict[str, Any] = {"id": t, "kind": "tool", "tool_name": t,
                                 "description": doc or ""}
        if roles:
            tnode["roles"] = sorted(roles)
        nodes.append(tnode)
        edges.append([AGENT, t])   # planner may call it
        edges.append([t, AGENT])   # its result feeds the next decision
    return {"name": name, "entry": AGENT, "nodes": nodes, "edges": edges, "source": "code"}


def flow_from_code(source: str, name: str = "code-agent") -> Dict[str, Any]:
    """Build a flow from agent source code (a single module's text).

    Prefers a FAITHFUL model: a LangGraph StateGraph (real nodes/edges + role per
    node from what the code does), then a scoped multi-agent crew, then a single
    AutoGen/AgentChat assistant with function-reference tools, then the worst-case
    tool-surface hub.
    """
    from .graph_extract import langgraph_flow
    g = langgraph_flow(source, name)
    if g and len(g["nodes"]) >= 2:
        return g
    crew = _crewai_flow(source, name)
    if crew and len(crew["nodes"]) >= 2:
        return crew
    ag = _agent_tools_flow(source, name)
    if ag and len(ag["nodes"]) >= 2:
        return ag
    return flow_from_code_tools(_extract_tools(source), name)


def flow_from_path(path: str, name: Optional[str] = None) -> Dict[str, Any]:
    """Import from a .py file or a directory of .py files (aggregated)."""
    sources: List[str] = []
    if os.path.isdir(path):
        for root, _dirs, files in os.walk(path):
            if any(part in ("node_modules", ".venv", "venv", "__pycache__", ".git") for part in root.split(os.sep)):
                continue
            for f in files:
                if f.endswith(".py"):
                    try:
                        with open(os.path.join(root, f)) as fh:
                            sources.append(fh.read())
                    except (OSError, UnicodeDecodeError):
                        continue
    else:
        try:
            with open(path, encoding="utf-8") as fh:
                sources.append(fh.read())
        except (OSError, UnicodeDecodeError) as e:
            raise ValueError(f"could not read {path}: {e}")

    tools: Dict[str, Dict[str, Any]] = {}
    for src in sources:
        try:
            for t in _extract_tools(src):
                tools.setdefault(t["name"], t)
        except SyntaxError:
            continue   # skip files that don't parse; best-effort by design
    return flow_from_code_tools(list(tools.values()), name or os.path.basename(path.rstrip("/")) or "code-agent")


def flow_from_code_tools(tools: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    """Build a flow from already-extracted tool dicts (shared by file/dir import)."""
    flow = flow_from_mcp({"tools": tools}, name=name)
    by_name = {t["name"]: t for t in tools}
    for n in flow["nodes"]:
        t = by_name.get(n.get("tool_name"))
        if t:
            for k in ("data_class", "exfil", "side_effect"):
                if k in t:
                    n[k] = t[k]
    flow["source"] = "code"
    return flow
