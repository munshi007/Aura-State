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
from dataclasses import dataclass, field
from typing import Dict, List


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
