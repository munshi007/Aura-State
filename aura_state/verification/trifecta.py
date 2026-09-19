"""Lethal-trifecta analysis: prove an agent can't be turned into an exfiltrator.

Simon Willison's *lethal trifecta* (https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/):
an LLM agent becomes an exfiltration weapon the moment three capabilities meet
on one reachable execution —

  1. access to **private data**       (a DB, files, secrets, an internal KB)
  2. exposure to **untrusted content** (a web page, an email body, a RAG chunk,
     a user-supplied document — anything an attacker can write into)
  3. the ability to **communicate externally** (send email, POST to a URL, a
     webhook, an upload — anything that carries bytes off the box)

With all three in scope, a prompt injection hidden in the untrusted content can
read the private data and ship it out. No amount of runtime output scanning
fixes it, because the model is doing exactly what it was told to do.

This module decides it *statically*, over the typed state machine, before the
agent ever runs. It reuses Aura's provenance model: taint from an untrusted
source that reaches an external sink **without crossing a sanitizer** is the
attacker's channel; a reachable private-data read is the payload. When both are
present and share scope, the trifecta is *closed* — and Aura reports the exact
three nodes and the path.

Classification of a tool's role from its name / declared side-effect is a
heuristic and is meant to be **overridden** — a node may state its own role via
``data_class`` (``"private"`` | ``"untrusted"`` | ``"public"``) and ``exfil``
(bool). Tools we cannot classify are surfaced as advisories, never silently
dropped: fail-closed means the human closes the gap, not the tool.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ── Role classification tables ────────────────────────────────────────────────
# Matched against "<tool_name> <description>" for tool nodes. Deliberately
# conservative and readable; override with `data_class` / `exfil` on the node.

# Stems allow suffixes (secret→secrets, ticket→tickets) via \w*; ambiguous
# short tokens (post, send) keep a trailing \b so `post` can't match `postgres`.
_UNTRUSTED_RX = re.compile(
    r"\b("
    r"https?|url|web|browse|scrape|crawl|fetch|download|"          # the open web
    r"rss|feed|"
    r"inbox|imap|email\w*read|readmail|"                           # inbound mail
    r"search|google|bing|serp|"                                    # web search
    r"user\w*(input|message|content|doc|upload)|attachment\w*|"    # user-supplied
    r"comment\w*|review\w*|ticket\w*|issue\w*|form\w*"             # third-party text
    r")", re.I)

_PRIVATE_RX = re.compile(
    r"\b("
    r"db|database\w*|sql\w*|postgres\w*|mysql|mongo\w*|sqlite|redis|quer(y|ies)|"  # datastores
    r"file\w*|fs|readfile|filesystem|"                            # local files
    r"secret\w*|vault|credential\w*|apikey|api_key|token\w*|password\w*|"  # secrets
    r"s3|storage|bucket\w*|blob\w*|"                              # object stores
    r"crm|salesforce|hubspot|"                                    # internal SaaS
    r"customer\w*|account\w*|record\w*|profile\w*|pii|"           # personal data
    r"kb|knowledge|internal|private|confidential|vectordb|embed\w*"  # internal KB / RAG
    r")", re.I)

_EXFIL_RX = re.compile(
    r"\b("
    r"send\w*|smtp|sendmail|"                                     # outbound mail
    r"post\b|put\b|patch\b|webhook\w*|"                           # outbound HTTP (post\b ≠ postgres)
    r"publish\w*|upload\w*|export\w*|"                            # push out
    r"payment\w*|charge\w*|stripe|transfer\w*|"                   # money
    r"slack|discord|telegram|sms|twilio|tweet\w*|notify\w*"       # messaging
    r")", re.I)


@dataclass
class TrifectaFinding:
    closed: bool                      # True = the trifecta is closed (vulnerable)
    untrusted: Optional[str]          # node exposing untrusted content
    private: Optional[str]            # node reading private data
    exfil: Optional[str]              # external-comms sink the taint reaches
    path: List[str] = field(default_factory=list)   # untrusted -> ... -> exfil
    shares_path: bool = False         # True = private read is an ancestor of the sink
    detail: str = ""


@dataclass
class TrifectaResult:
    verified: bool                    # True iff no trifecta is closed
    findings: List[TrifectaFinding] = field(default_factory=list)
    unclassified: List[str] = field(default_factory=list)   # read-tools we couldn't tag
    roles: Dict[str, List[str]] = field(default_factory=dict)  # node -> [roles], for reporting

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified": self.verified,
            "unclassified": self.unclassified,
            "roles": self.roles,
            "findings": [f.__dict__ for f in self.findings],
        }


def classify_roles(n: Dict[str, Any]) -> Tuple[Set[str], bool]:
    """Return (roles, is_unclassified_read).

    roles ⊆ {"untrusted", "private", "exfil", "sanitizer"}. Explicit node fields
    (``capability``, ``data_class``, ``exfil``, sanitizer kind) always win over
    the name heuristic. ``is_unclassified_read`` flags a read-side tool whose
    role we could not infer — reported as an advisory, not dropped.
    """
    roles: Set[str] = set()
    kind = n.get("kind") or n.get("type") or "extract"
    cap = n.get("capability")
    se = n.get("side_effect")
    dc = n.get("data_class")

    if kind == "sanitizer" or cap == "sanitizer":
        roles.add("sanitizer")

    # Explicit overrides.
    if n.get("exfil") is True:
        roles.add("exfil")
    if dc == "untrusted":
        roles.add("untrusted")
    elif dc == "private":
        roles.add("private")
    if cap == "untrusted":
        roles.add("untrusted")
    if cap == "sink":
        roles.add("exfil")

    unclassified = False
    if kind == "tool":
        blob = f"{n.get('tool_name') or n.get('id') or ''} {n.get('description') or n.get('system_prompt') or ''}"
        if se in ("write", "external") or _EXFIL_RX.search(blob):
            roles.add("exfil")
        if se == "read" or se is None:
            hit = False
            if _UNTRUSTED_RX.search(blob):
                roles.add("untrusted"); hit = True
            if _PRIVATE_RX.search(blob):
                roles.add("private"); hit = True
            # a pure read tool we couldn't place is a real blind spot
            if not hit and "exfil" not in roles and dc != "public":
                unclassified = True
    return roles, unclassified


def _reachable_from(entry: str, adj: Dict[str, List[str]], nodes: Set[str]) -> Set[str]:
    if entry not in nodes:
        return set(nodes)  # unknown entry -> treat all as reachable (fail-closed)
    seen = {entry}
    q = deque([entry])
    while q:
        x = q.popleft()
        for t in adj.get(x, []):
            if t in nodes and t not in seen:
                seen.add(t); q.append(t)
    return seen


def _ancestors(target: str, radj: Dict[str, List[str]], nodes: Set[str]) -> Set[str]:
    """Nodes that can reach `target` (i.e. may execute before it on a path)."""
    seen = {target}
    q = deque([target])
    while q:
        x = q.popleft()
        for p in radj.get(x, []):
            if p in nodes and p not in seen:
                seen.add(p); q.append(p)
    return seen


def analyze_trifecta(nodes: List[Dict[str, Any]], edges: List[List[str]],
                     entry: Optional[str] = None) -> TrifectaResult:
    """Decide whether the lethal trifecta can close on any reachable execution.

    Sound (may-reach): reports the trifecta closed if *some* path lets untrusted
    content reach an external sink unsanitized while private data is in scope.
    """
    ids = {n["id"] for n in nodes}
    by_id = {n["id"]: n for n in nodes}
    adj: Dict[str, List[str]] = {i: [] for i in ids}
    radj: Dict[str, List[str]] = {i: [] for i in ids}
    for a, b in edges:
        if a in ids and b in ids:
            adj[a].append(b); radj[b].append(a)

    entry = entry or (nodes[0]["id"] if nodes else None)
    reach = _reachable_from(entry, adj, ids) if entry else set(ids)

    roles_map: Dict[str, Set[str]] = {}
    unclassified: List[str] = []
    for n in nodes:
        r, unk = classify_roles(n)
        roles_map[n["id"]] = r
        if unk and n["id"] in reach:
            unclassified.append(n["id"])

    untrusted = {i for i in reach if "untrusted" in roles_map[i]}
    private = {i for i in reach if "private" in roles_map[i]}
    exfil = {i for i in reach if "exfil" in roles_map[i]}
    sanitizers = {i for i in ids if "sanitizer" in roles_map[i]}

    # Untrusted taint reaching an exfil sink without crossing a sanitizer.
    # For each untrusted source, DFS; a sanitizer prunes; an exfil node hit while
    # tainted is the attacker's channel. Record (exfil -> (source, path)).
    tainted_exfil: Dict[str, Tuple[str, List[str]]] = {}
    for src in untrusted:
        stack = [(src, [src])]
        seen = set()
        while stack:
            node, path = stack.pop()
            if node != src and node in sanitizers:
                continue                       # taint cleaned here
            if node != src and node in exfil and node not in tainted_exfil:
                tainted_exfil[node] = (src, path)
                # keep exploring past it in case of further sinks on other paths
            if node in seen:
                continue
            seen.add(node)
            for t in adj.get(node, []):
                if t in ids:
                    stack.append((t, path + [t]))

    findings: List[TrifectaFinding] = []
    for sink, (src, path) in tainted_exfil.items():
        if not private:
            continue                            # injection path, but no private data -> not a full trifecta
        anc = _ancestors(sink, radj, ids)
        on_path = sorted(private & anc)         # private read that executes before the sink
        p_node = on_path[0] if on_path else sorted(private)[0]
        shares = bool(on_path)
        detail = (
            f"prompt injection at '{src}' (untrusted) can reach external sink "
            f"'{sink}' unsanitized while '{p_node}' brings private data into scope"
            + ("" if shares else " (shared via agent memory, not the same path)")
        )
        findings.append(TrifectaFinding(
            closed=True, untrusted=src, private=p_node, exfil=sink,
            path=path, shares_path=shares, detail=detail))

    return TrifectaResult(
        verified=not findings,
        findings=findings,
        unclassified=sorted(set(unclassified)),
        roles={i: sorted(r) for i, r in roles_map.items() if r},
    )
