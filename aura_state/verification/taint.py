"""
Capability-typed dataflow: prove untrusted data can't reach a dangerous sink.

Static taint analysis over the typed node graph (the same shape Agentproof,
arXiv:2603.20356, verifies). Nodes are labelled with provenance/capability:

- ``untrusted_source``: the node emits untrusted data (e.g. an LLM extraction
  or an external tool result). This is where taint originates.
- ``dangerous_sink``: the node performs an irreversible / side-effecting action
  (send email, write to an ERP, spend money). Taint must never reach it.
- ``sanitizer``: the node validates/constrains the data so downstream is clean;
  taint does not propagate past it.

The analysis propagates taint forward along declared transitions (a fixpoint
over tainted reachability) and reports any path from an untrusted source to a
dangerous sink that does not cross a sanitizer. Because it tracks *provenance,
not content*, it is immune to the encoding tricks that defeat runtime scanners,
and because it runs at design time over the graph, the verdict compiles into
the AuraContract alongside the CTL properties.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class TaintViolation:
    source: str                 # untrusted-source node where the taint originates
    sink: str                   # dangerous sink it can reach
    path: List[str]             # a concrete tainted path source -> ... -> sink


@dataclass
class TaintResult:
    verified: bool              # True iff no untrusted source can reach a dangerous sink
    violations: List[TaintViolation] = field(default_factory=list)


def _flag(node, name: str) -> bool:
    return bool(getattr(node, name, False))


def analyze_taint(engine) -> TaintResult:
    """Prove no untrusted source reaches a dangerous sink without a sanitizer.

    Sound (may-reach) analysis: if *any* tainted path reaches a sink, it is a
    violation. Sanitizers prune propagation. Returns every violating
    source→sink path found (a real counterexample for repair / the contract).
    """
    nodes = engine._nodes
    transitions = engine._transitions

    sources = [n for n, nd in nodes.items() if _flag(nd, "untrusted_source")]
    violations: List[TaintViolation] = []

    for src in sources:
        # DFS from the source carrying the concrete path. The source itself is
        # tainted (it emits untrusted data); a sanitizer stops propagation; a
        # dangerous sink reached while tainted is a violation.
        stack = [(src, [src])]
        seen = set()
        while stack:
            node, path = stack.pop()
            nd = nodes[node]

            if node != src and _flag(nd, "dangerous_sink"):
                violations.append(TaintViolation(source=src, sink=node, path=path))
                continue  # record; don't propagate past the sink on this path

            if node != src and _flag(nd, "sanitizer"):
                continue  # taint cleaned here -> downstream is safe

            if node in seen:
                continue
            seen.add(node)

            for tgt in transitions.get(node, []):
                if tgt in nodes:
                    stack.append((tgt, path + [tgt]))

    return TaintResult(verified=not violations, violations=violations)


# ── Value/field-level taint (full 0014) ──────────────────────────────────────
#
# Node-level taint asks "can any untrusted node reach a sink?" Field-level taint
# is precise: it tracks WHICH named fields carry untrusted provenance as they
# flow through the graph, so a clean field passes through a sink untouched while
# only a *tainted* field reaching that sink is a violation. This is CaMeL's
# per-value capability model, computed statically over the typed graph.
#
# The taint state at a node is a map {field_name -> origin_node}. The wildcard
# field "*" means "all fields tainted" (used when an untrusted node has no
# declared schema). The transfer function per node is:
#     out(n) = introduced(n)  ∪  ( in(n) \ sanitized(n) )
# computed to a fixpoint (monotone -> converges even with cycles).

ALL = "*"


@dataclass
class FieldTaintViolation:
    field: str                  # the tainted field that reaches the sink ("*" = any)
    source: str                 # node that introduced the taint on that field
    sink: str                   # dangerous sink node it reaches
    path: List[str] = field(default_factory=list)


@dataclass
class FieldTaintResult:
    verified: bool
    violations: List[FieldTaintViolation] = field(default_factory=list)


def _introduced(node) -> Set[str]:
    explicit = getattr(node, "untrusted_fields", None)
    if explicit:
        return set(explicit)
    if getattr(node, "untrusted_source", False):
        schema = getattr(node, "extracts", None)
        if schema is not None:
            return set(schema.model_fields.keys())   # LLM-extracted fields are untrusted
        return {ALL}
    return set()


def _sink_fields(node) -> Set[str]:
    explicit = getattr(node, "sink_fields", None)
    if explicit:
        return set(explicit)
    if getattr(node, "dangerous_sink", False):
        return {ALL}
    return set()


def _sanitized(node) -> Set[str]:
    explicit = getattr(node, "sanitizes_fields", None)
    if explicit:
        return set(explicit)
    if getattr(node, "sanitizer", False):
        return {ALL}
    return set()


def _bfs_path(transitions, src, dst) -> List[str]:
    if src == dst:
        return [src]
    q = deque([(src, [src])])
    seen = {src}
    while q:
        node, path = q.popleft()
        for tgt in transitions.get(node, []):
            if tgt == dst:
                return path + [tgt]
            if tgt not in seen:
                seen.add(tgt)
                q.append((tgt, path + [tgt]))
    return [src, dst]


def analyze_field_taint(engine) -> FieldTaintResult:
    """Field-level static taint: prove no *tainted field* reaches a dangerous sink.

    Precise over node-level: distinguishes which fields are untrusted, lets a
    sanitizer clear specific fields, and attributes each violation to the field
    and the node that introduced it.
    """
    nodes = engine._nodes
    transitions = engine._transitions
    preds: Dict[str, List[str]] = {n: [] for n in nodes}
    for src, tgts in transitions.items():
        for t in tgts:
            if t in preds:
                preds[t].append(src)

    # Fixpoint on out(n): {field -> origin}
    out: Dict[str, Dict[str, str]] = {n: {} for n in nodes}
    changed = True
    while changed:
        changed = False
        for n in nodes:
            incoming: Dict[str, str] = {}
            for p in preds[n]:
                for f, orig in out[p].items():
                    incoming.setdefault(f, orig)
            san = _sanitized(nodes[n])
            if ALL in san:
                kept: Dict[str, str] = {}
            else:
                kept = {f: o for f, o in incoming.items() if f not in san}
            new_out = dict(kept)
            for f in _introduced(nodes[n]):
                new_out.setdefault(f, n)
            if new_out != out[n]:
                out[n] = new_out
                changed = True

    # Violations: a tainted field arriving at a dangerous sink.
    violations: List[FieldTaintViolation] = []
    for n in nodes:
        sinkf = _sink_fields(nodes[n])
        if not sinkf:
            continue
        incoming: Dict[str, str] = {}
        for p in preds[n]:
            for f, orig in out[p].items():
                incoming.setdefault(f, orig)
        for f, orig in incoming.items():
            hit = (ALL in sinkf) or (f in sinkf) or (f == ALL)
            if hit:
                violations.append(FieldTaintViolation(
                    field=f, source=orig, sink=n, path=_bfs_path(transitions, orig, n),
                ))
    return FieldTaintResult(verified=not violations, violations=violations)
