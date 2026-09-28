"""Information-flow control over an agent graph — a non-interference check.

This unifies the two passes that used to disagree. The lethal-trifecta pass
already classifies node *roles* (`classify_roles`: untrusted / private / exfil /
sanitizer) and walks untrusted→exfil with sanitizer pruning. The old taint pass
used a *different*, capability-only model (a tool's side-effect), so a tool whose
untrusted nature was known by name/`data_class`/`roles` — but not by an explicit
`capability` — was **not** treated as a taint source: a fail-open the trifecta
pass did not share.

`analyze_ifc` runs the **integrity** half of information-flow control on the SAME
role model as the trifecta: untrusted (low-integrity) data must not reach a
*consequential* sink (an external send or a local write) without passing a
sanitizer (a declassifier). That is exactly the injection-safety property — a
prompt injection in untrusted content must not be able to drive a consequential
action — and it is now decided consistently with the confidentiality half
(private→exfil) that `analyze_trifecta` decides. Sound / fail-closed: an
unknown-capability node is surfaced, never silently trusted.

Refs: information-flow control for agents — AgentFlow (arXiv:2608.22868) and
FIDES (arXiv:2505.23643) formalize a confidentiality×integrity label lattice;
this is the static, import-time integrity check on our role model.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .trifecta import classify_roles


@dataclass
class IntegrityFlow:
    source: str          # untrusted (low-integrity) node
    sink: str            # consequential sink the taint reaches
    external: bool       # True = external send (exfil), False = local write
    path: List[str] = field(default_factory=list)


@dataclass
class IFCResult:
    verified: bool                       # True iff no untrusted→consequential-sink flow
    flows: List[IntegrityFlow] = field(default_factory=list)
    unclassified: List[str] = field(default_factory=list)


def _reachable_from(entry: Optional[str], adj: Dict[str, List[str]], ids: Set[str]) -> Set[str]:
    if not entry or entry not in ids:
        return set(ids)                  # unknown entry -> all reachable (fail-closed)
    seen = {entry}
    q = deque([entry])
    while q:
        x = q.popleft()
        for t in adj.get(x, []):
            if t in ids and t not in seen:
                seen.add(t); q.append(t)
    return seen


def analyze_ifc(nodes: List[Dict[str, Any]], edges: List[List[str]],
                entry: Optional[str] = None) -> IFCResult:
    ids = {n["id"] for n in nodes}
    by_id = {n["id"]: n for n in nodes}
    adj: Dict[str, List[str]] = {i: [] for i in ids}
    for a, b in edges:
        if a in ids and b in ids:
            adj[a].append(b)

    entry = entry or (nodes[0]["id"] if nodes else None)
    reach = _reachable_from(entry, adj, ids)

    roles: Dict[str, Set[str]] = {}
    unclassified: List[str] = []
    for n in nodes:
        r, unk = classify_roles(n)
        roles[n["id"]] = r
        if unk and n["id"] in reach:
            unclassified.append(n["id"])

    untrusted = {i for i in reach if "untrusted" in roles[i]}
    sanitizers = {i for i in ids if "sanitizer" in roles[i]}

    def is_sink(i: str) -> Optional[bool]:
        """None = not a consequential sink; True = external (exfil); False = local write."""
        if "exfil" in roles[i]:
            return True
        n = by_id[i]
        kind = n.get("kind") or n.get("type")
        se = n.get("side_effect")
        if kind == "tool" and se == "external":
            return True
        if kind == "tool" and se == "write":
            return False
        return None

    # Integrity: from each untrusted source, walk forward; a sanitizer declassifies
    # (prune); a consequential sink hit while still tainted is a non-interference
    # violation (untrusted data can drive that action). One flow per (source, sink).
    seen_pairs: Set[str] = set()
    flows: List[IntegrityFlow] = []
    for src in sorted(untrusted):
        stack = [(src, [src])]
        visited: Set[str] = set()
        while stack:
            node, path = stack.pop()
            if node != src and node in sanitizers:
                continue                 # taint cleared here
            ext = is_sink(node)
            # A node that is BOTH the untrusted source and a consequential sink is a
            # self-contained channel — one tool that ingests untrusted input and acts
            # on it externally / writes. Flag it too (mirrors trifecta's node==src
            # case); guarding this on `node != src` was a fail-open.
            if ext is not None:
                pk = f"{src}->{node}"
                if pk not in seen_pairs:
                    seen_pairs.add(pk)
                    flows.append(IntegrityFlow(source=src, sink=node, external=ext, path=path))
                # keep walking past a sink in case further sinks lie downstream
            for t in adj.get(node, []):
                if t not in visited:
                    visited.add(t); stack.append((t, path + [t]))

    return IFCResult(verified=not flows, flows=flows, unclassified=unclassified)
