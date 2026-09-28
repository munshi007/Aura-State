"""Capability containment / least-privilege over an agent graph.

The majority of enterprise AI-agent incidents involve **no attacker** — the agent
is over-permissioned and takes an unintended consequential action on its way to a
task (data deletion, an unexpected write, a payment). This is the design-time
check for that class: given a declared **capability manifest** (what the agent is
*allowed* to do), prove that every effect it can reach stays inside it — a
tool-call *envelope* check. If the graph can reach a tool / side-effect / data
class the manifest does not permit, that reachable over-reach is reported.

It's opt-in: with no manifest there is no declared scope to contain, so no
finding is produced (the summary says "not declared") — declaring a tighter
scope than the agent's raw capabilities is the whole point.

Refs: capability-containment via refinement-type tool-call envelopes
(arXiv:2605.23951); least-privilege systems Progent / MiniScope. This is the
static, import-time version on our node model.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ContainmentResult:
    verified: bool
    violations: List[Tuple[str, str, str]] = field(default_factory=list)  # (node, dimension, value)
    declared: bool = True   # False = no manifest given (nothing to contain)


def _reachable(entry: Optional[str], adj: Dict[str, List[str]], ids: Set[str]) -> Set[str]:
    if not entry or entry not in ids:
        return set(ids)                 # unknown entry -> all reachable (fail-closed)
    seen = {entry}
    q = deque([entry])
    while q:
        x = q.popleft()
        for t in adj.get(x, []):
            if t in ids and t not in seen:
                seen.add(t); q.append(t)
    return seen


def analyze_containment(nodes: List[Dict[str, Any]], edges: List[List[str]],
                        entry: Optional[str], manifest: Optional[Dict[str, Any]]) -> ContainmentResult:
    """Prove reachable effects ⊆ the declared manifest.

    manifest keys (each optional — only a key that is present is enforced):
      - ``side_effects``: allowed values in {read, write, external}
      - ``tools``: allowed tool_name allowlist
      - ``data_classes``: allowed data_class allowlist
    Accepts either a flat manifest or one nested under ``allow``.
    """
    if not isinstance(manifest, dict):
        return ContainmentResult(verified=True, declared=False)
    _nested = manifest.get("allow")
    # a non-empty nested `allow` wins; an empty/absent one falls back to flat keys
    # (mixing `{"allow": {}, "side_effects": [...]}` used to silently drop the allowlist)
    allow = _nested if (isinstance(_nested, dict) and _nested) else manifest

    def _set(key: str) -> Optional[Set[str]]:
        v = allow.get(key)
        return set(v) if isinstance(v, (list, tuple, set)) else None

    allow_se, allow_tools, allow_dc = _set("side_effects"), _set("tools"), _set("data_classes")
    if allow_se is None and allow_tools is None and allow_dc is None:
        return ContainmentResult(verified=True, declared=False)  # empty manifest = nothing declared

    ids = {n["id"] for n in nodes}
    adj: Dict[str, List[str]] = {i: [] for i in ids}
    for a, b in edges:
        if a in ids and b in ids:
            adj[a].append(b)
    reach = _reachable(entry or (nodes[0]["id"] if nodes else None), adj, ids)

    violations: List[Tuple[str, str, str]] = []
    for n in nodes:
        if n["id"] not in reach:
            continue
        if (n.get("kind") or n.get("type")) != "tool":
            continue
        se = n.get("side_effect")
        # an exfil-classified tool (by `exfil` flag or `roles`) IS an external effect,
        # even if `side_effect` is unset — otherwise it escapes a read-only manifest.
        if se is None and (n.get("exfil") is True or "exfil" in (n.get("roles") or [])):
            se = "external"
        tn, dc = n.get("tool_name"), n.get("data_class")
        if allow_se is not None and se is not None and se not in allow_se:
            violations.append((n["id"], "side_effect", se))
        if allow_tools is not None and tn is not None and tn not in allow_tools:
            violations.append((n["id"], "tool", tn))
        if allow_dc is not None and dc is not None and dc not in allow_dc:
            violations.append((n["id"], "data_class", dc))

    return ContainmentResult(verified=not violations, violations=violations, declared=True)
