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


def flow_from_code(source: str, name: str = "code-agent") -> Dict[str, Any]:
    """Build a hub flow from agent source code (a single module's text)."""
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
        with open(path) as fh:
            sources.append(fh.read())

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
