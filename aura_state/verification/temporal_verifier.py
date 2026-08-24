"""
Temporal logic verification for Aura-State workflows.

Compiles an AuraEngine's node graph into a Kripke structure and
verifies CTL properties using pyModelChecking.

Two correctness invariants govern this module (task 0005):

1. **Deadlock detection is STRUCTURAL, not a CTL formula.** CTL semantics
   require a *total* transition relation (pyModelChecking raises otherwise), so
   `compile_kripke` totalizes the graph by adding self-loops to every sink. That
   totalization is precisely what would *erase* an accidental dead-end: a sink
   with no way out becomes a state that loops forever. We therefore detect
   dead-ends by scanning the ORIGINAL, pre-totalized graph (`find_dead_ends` /
   `no_dead_ends`) BEFORE any self-loop is added, and totalize only afterwards
   for CTL well-formedness. Intended terminals and accidental sinks are labeled
   with distinct atomic props ("terminal" vs "dead_end") so the two are never
   conflated.

2. **Reachability / eventuality are judged at the INIT state, not every state.**
   A CTL state formula is evaluated at a single state; for a workflow the
   meaningful one is the entry node. EF/AF checked over *all* states spuriously
   report VIOLATED because a reachable target legitimately fails the property at
   states from which it is not reachable. See `verify_property`.
"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Set

from pyModelChecking import Kripke
from pyModelChecking.CTL import modelcheck, A, E, G, F, X, U, Not, And, Or, Imply

logger = logging.getLogger("aura_state.verification")


class PropertyResult(Enum):
    PROVEN = "proven"
    VIOLATED = "violated"


@dataclass
class VerificationResult:
    property_text: str
    formula_repr: str
    result: PropertyResult
    satisfying_states: Set[str]
    violating_states: Set[str]


# ---------------------------------------------------------------------------
# Structural graph helpers (operate on the ORIGINAL, pre-totalized graph)
# ---------------------------------------------------------------------------

def _find_sinks(nodes: Dict, transitions: Dict) -> Set[str]:
    """Nodes with no outgoing edge in the ORIGINAL graph (before totalization)."""
    return {name for name in nodes.keys() if not transitions.get(name)}


def find_dead_ends(
    nodes: Dict, transitions: Dict, terminals: Optional[Iterable[str]] = None
) -> Set[str]:
    """Return accidental dead-ends: sinks that are NOT declared intended terminals.

    Structural check over the ORIGINAL pre-totalized graph. This MUST run before
    (or independently of) `compile_kripke`'s totalization — once self-loops are
    added for CTL totality, every sink has a successor and no dead-end is
    findable. `terminals=None` means "every sink is intended" (no dead-ends),
    which is the backward-compatible default.
    """
    if terminals is None:
        return set()
    sinks = _find_sinks(nodes, transitions)
    return sinks - set(terminals)


def _find_init_node(
    nodes: Dict, transitions: Dict, init_node: Optional[str] = None
) -> str:
    """Identify the workflow entry/init node.

    Explicit `init_node` wins. Otherwise the entry is a graph source (a node with
    no incoming edge); if several exist, the first-registered one is chosen. If
    the graph is cyclic with no source, fall back to the first registered node.
    The engine's start node is whatever it is first driven from, and registration
    order records that entry.
    """
    if init_node is not None:
        return init_node
    has_incoming: Set[str] = set()
    for _src, targets in transitions.items():
        for tgt in targets:
            has_incoming.add(tgt)
    sources = [name for name in nodes.keys() if name not in has_incoming]
    if sources:
        return sources[0]
    return next(iter(nodes.keys()))


def _reachable_from(init_node: str, edges: List) -> Set[str]:
    """States reachable from `init_node` following `edges` (a list of (a, b))."""
    adj: Dict[str, List[str]] = {}
    for a, b in edges:
        adj.setdefault(str(a), []).append(str(b))
    seen = {init_node}
    stack = [init_node]
    while stack:
        cur = stack.pop()
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def compile_kripke(
    nodes: Dict, transitions: Dict, terminals: Optional[Iterable[str]] = None
) -> Kripke:
    """Build a *totalized* Kripke structure from the engine's nodes and edges.

    Totalization (self-looping every sink) is done ONLY to satisfy CTL's
    requirement of a total transition relation — pyModelChecking raises on a
    non-total relation. It is deliberately kept SEPARATE from deadlock detection
    (`find_dead_ends`), which scans the original pre-totalized graph; self-looping
    here would otherwise mask the accidental dead-ends we need to find.

    Labeling distinguishes intent from structure:
      * a sink declared an intended terminal  -> atomic prop ``"terminal"``
      * a sink NOT declared a terminal (accidental) -> atomic prop ``"dead_end"``

    `terminals=None` is the backward-compatible default: every sink is treated as
    an intended terminal.
    """
    if terminals is None:
        terminals_set = _find_sinks(nodes, transitions)
    else:
        terminals_set = set(terminals)

    states = list(nodes.keys())
    edges = []
    labels: Dict[str, Set[str]] = {}

    for node_name, node_obj in nodes.items():
        props: Set[str] = {node_name}

        if node_obj.extracts:
            props.add("has_extraction")
        if node_obj.sandbox_rule:
            props.add("has_sandbox")
        if node_obj.consensus > 1:
            props.add("has_consensus")

        targets = transitions.get(node_name, [])
        for target in targets:
            edges.append((node_name, target))

        if not targets:
            # Structural sink. Label by INTENT (distinct props), then totalize
            # with a self-loop purely for CTL well-formedness.
            if node_name in terminals_set:
                props.add("terminal")
            else:
                props.add("dead_end")
            edges.append((node_name, node_name))

        labels[node_name] = props

    return Kripke(S=states, R=edges, L=labels)


# -- Built-in CTL property constructors --

def reachability(target_node: str):
    """EF(target_node) — target is reachable from the evaluation state.

    Judged at the init state by `verify_property`: PROVEN iff the target is
    reachable from the workflow entry.
    """
    return E(F(target_node))


def always_before(before_node: str, after_node: str):
    """Ordering: on every path, ``before`` holds before ``after`` ever occurs.

    CTL: ``¬E[¬before U (after ∧ ¬before)]``.
    Read: "there is NO path along which we reach an ``after`` state while
    ``before`` has not yet held, passing only through ¬before states." If no such
    witnessing path exists, then on every path ``before`` necessarily holds at or
    before the first ``after`` — i.e. genuine temporal ordering.

    This replaces the previous ``AG(after ⇒ before)``, which only tested whether
    the two labels *co-occur in a single state* and said nothing about ordering.
    """
    return Not(E(U(Not(before_node), And(after_node, Not(before_node)))))


def mutual_exclusion(node_a: str, node_b: str):
    """AG(¬(node_a ∧ node_b)) — can never be in both states simultaneously.

    Judged at the init state, AG ranges over exactly the states reachable from
    the entry, which is the correct scope for a universal-safety property.
    """
    return A(G(Not(And(node_a, node_b))))


def eventual_completion(*terminal_nodes: str):
    """AF(terminal_1 ∨ ...) — every path eventually reaches a terminal.

    Judged at the init state: PROVEN iff every path FROM the entry reaches a
    terminal.
    """
    if len(terminal_nodes) == 1:
        return A(F(terminal_nodes[0]))
    combined = Or(*terminal_nodes)
    return A(F(combined))


def no_dead_ends(
    nodes: Dict, transitions: Dict, terminals: Optional[Iterable[str]] = None
) -> "VerificationResult":
    """STRUCTURAL check: no ACCIDENTAL dead-ends in the workflow.

    "No accidental dead-ends" = every non-terminal node has at least one outgoing
    edge in the ORIGINAL graph. This is intentionally NOT a CTL formula: the
    Kripke structure must be totalized (self-loops on sinks) for CTL to run, and
    that totalization turns every dead-end into a self-looping state, erasing it.
    So we scan the pre-totalized graph directly.

    The old implementation returned ``A(G(Not("terminal")))`` — which asserts NO
    state is ever terminal and so fails on every healthy workflow (reversed).

    Returns a VerificationResult: VIOLATED with the accidental sinks in
    ``violating_states`` if any exist, else PROVEN.
    """
    dead = find_dead_ends(nodes, transitions, terminals)
    node_names = set(nodes.keys())
    result = PropertyResult.VIOLATED if dead else PropertyResult.PROVEN
    return VerificationResult(
        property_text="",
        formula_repr="STRUCTURAL: every non-terminal node has a successor",
        result=result,
        satisfying_states=node_names - dead,
        violating_states=set(dead),
    )


def verify_property(
    kripke: Kripke, formula, all_states: List[str], init_node: str
) -> VerificationResult:
    """Model-check `formula` and decide PROVEN/VIOLATED **at the init state**.

    A CTL state formula is evaluated at a single state. For a workflow the
    meaningful state is the entry/init node, so the decision is:

        PROVEN iff ``init_node`` is in the satisfying set.

    Why not "every state must satisfy"?
      * EF(target)/reachability: only states from which the target is reachable
        satisfy it. A perfectly reachable target still fails at sibling states,
        so "all states" would spuriously report VIOLATED. The question is whether
        the ENTRY can reach the target.
      * AF(terminal)/eventual_completion: PROVEN iff every path FROM the entry
        reaches a terminal.
      * AG(safety)/mutual_exclusion/always_before: AG evaluated at the init state
        already ranges over exactly the states REACHABLE from init — the correct
        scope for a universal-safety property (unreachable states are
        irrelevant). ``violating_states`` is reported over that reachable set.
    """
    satisfying = modelcheck(kripke, formula)
    satisfying_names = {str(s) for s in satisfying}

    reachable = _reachable_from(init_node, list(kripke.transitions()))
    violating = reachable - satisfying_names

    result = (
        PropertyResult.PROVEN
        if init_node in satisfying_names
        else PropertyResult.VIOLATED
    )

    return VerificationResult(
        property_text="",
        formula_repr=str(formula),
        result=result,
        satisfying_states=satisfying_names,
        violating_states=violating,
    )


def verify_engine(
    engine,
    properties: List[dict],
    terminals: Optional[Iterable[str]] = None,
    init_node: Optional[str] = None,
) -> List[VerificationResult]:
    """
    Verify a list of CTL properties against an AuraEngine's workflow graph.

    Each property is a dict with:
        - "description": human-readable string
        - "formula": a CTL formula object (from the constructors above)

    `terminals` declares which sink nodes are intended terminals (defaults to
    "every sink"); `init_node` overrides the auto-detected entry node. All CTL
    properties are judged at the init state (see `verify_property`).

    Returns a list of VerificationResults.
    """
    kripke = compile_kripke(engine._nodes, engine._transitions, terminals=terminals)
    all_states = list(engine._nodes.keys())
    init = _find_init_node(engine._nodes, engine._transitions, init_node)
    results = []

    for prop in properties:
        vr = verify_property(kripke, prop["formula"], all_states, init)
        vr.property_text = prop.get("description", "")
        results.append(vr)

        status = "PROVEN" if vr.result == PropertyResult.PROVEN else "VIOLATED"
        logger.info(f"[Verify] {status}: {vr.property_text} (init={init})")
        if vr.violating_states:
            logger.warning(f"  Violating (reachable) states: {vr.violating_states}")

    return results
