"""
Temporal logic verification for Aura-State workflows.

Compiles an AuraEngine's node graph into a Kripke structure and
verifies CTL properties using pyModelChecking.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Type
from enum import Enum

from pyModelChecking import Kripke
from pyModelChecking.CTL import modelcheck, A, E, G, F, X, Not, And, Or, Imply

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


def compile_kripke(nodes: Dict, transitions: Dict) -> Kripke:
    """Build a Kripke structure from the engine's registered nodes and edges."""
    states = list(nodes.keys())
    edges = []
    labels = {}

    for node_name, node_obj in nodes.items():
        props = set()
        props.add(node_name)

        if node_obj.extracts:
            props.add("has_extraction")
        if node_obj.sandbox_rule:
            props.add("has_sandbox")
        if node_obj.consensus > 1:
            props.add("has_consensus")

        labels[node_name] = props

        targets = transitions.get(node_name, [])
        for target in targets:
            edges.append((node_name, target))

        if not targets:
            props.add("terminal")
            edges.append((node_name, node_name))

    return Kripke(S=states, R=edges, L=labels)


# -- Built-in CTL property constructors --

def reachability(target_node: str):
    """EF(target_node) — target is reachable from at least one path."""
    return E(F(target_node))


def always_before(before_node: str, after_node: str):
    """Approximation: AG(¬after_node) unless before_node was on the path.
    Encoded as: there is no path where after_node is reached without before_node.
    ¬EF(after_node ∧ ¬before_node) — no state satisfies after without before's label.
    Note: This is a label-level check since CTL operates on state labels.
    """
    return A(G(Imply(after_node, before_node)))


def mutual_exclusion(node_a: str, node_b: str):
    """AG(¬(node_a ∧ node_b)) — can never be in both states simultaneously."""
    return A(G(Not(And(node_a, node_b))))


def eventual_completion(*terminal_nodes: str):
    """AF(terminal_1 ∨ terminal_2 ∨ ...) — every path eventually reaches a terminal."""
    if len(terminal_nodes) == 1:
        return A(F(terminal_nodes[0]))
    combined = Or(*terminal_nodes)
    return A(F(combined))


def no_dead_ends():
    """AG(¬terminal) is false if terminals exist, so instead:
    Check that every non-terminal node has at least one successor.
    This is structural and handled during Kripke compilation."""
    return A(G(Not("terminal")))


def verify_property(kripke: Kripke, formula, all_states: List[str]) -> VerificationResult:
    """Run CTL model checking and return which states satisfy the formula."""
    satisfying = modelcheck(kripke, formula)
    satisfying_names = {str(s) for s in satisfying}
    all_names = set(all_states)
    violating = all_names - satisfying_names

    result = PropertyResult.VIOLATED if violating else PropertyResult.PROVEN

    return VerificationResult(
        property_text="",
        formula_repr=str(formula),
        result=result,
        satisfying_states=satisfying_names,
        violating_states=violating,
    )


def verify_engine(engine, properties: List[dict]) -> List[VerificationResult]:
    """
    Verify a list of CTL properties against an AuraEngine's workflow graph.

    Each property is a dict with:
        - "description": human-readable string
        - "formula": a CTL formula object (from the constructors above)

    Returns a list of VerificationResults.
    """
    kripke = compile_kripke(engine._nodes, engine._transitions)
    all_states = list(engine._nodes.keys())
    results = []

    for prop in properties:
        vr = verify_property(kripke, prop["formula"], all_states)
        vr.property_text = prop.get("description", "")
        results.append(vr)

        status = "PROVEN" if vr.result == PropertyResult.PROVEN else "VIOLATED"
        logger.info(f"[Verify] {status}: {vr.property_text}")
        if vr.violating_states:
            logger.warning(f"  Violating states: {vr.violating_states}")

    return results
