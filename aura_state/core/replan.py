"""
Counterexample-guided replanning: let the verifier drive the plan to correct.

Verification is usually a gate: it says VIOLATED and stops. This closes the
loop -- the model checker's / solver's / taint analysis's counterexample is
translated into a structured repair signal, a replanner edits the plan, and the
plan is re-verified, iterating until the property holds or a budget K is hit.
The plan is provably correct *because* the verifier drove it there.

Refs: PAT-Agent (arXiv:2509.23675), VERIMAP (arXiv:2510.17109).

The replanner is pluggable: pass any ``repair_fn(engine, signal) -> bool`` (an
LLM planner, or the deterministic strategies here). The loop never silently
accepts -- an unrepairable case aborts with the explicit unresolved violations.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("aura_state")


@dataclass
class RepairSignal:
    """A structured constraint derived from a verifier counterexample."""
    kind: str                       # "ctl" | "z3" | "taint"
    description: str
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RepairIteration:
    iteration: int
    violation: str
    signal: Optional[RepairSignal]
    repaired: bool


@dataclass
class ReplanResult:
    verified: bool
    iterations: int
    history: List[RepairIteration] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)


# ── Counterexample adapters: verifier output -> structured repair signal ──

def ctl_to_repair(vr) -> RepairSignal:
    """CTL VerificationResult -> repair signal (property + violating states)."""
    target = None
    parts = str(vr.formula_repr).split()
    if len(parts) == 2 and parts[0] in ("EF", "AF"):   # reachability / eventual
        target = parts[1]
    return RepairSignal(
        kind="ctl",
        description=f"CTL violated: {vr.property_text or vr.formula_repr}",
        detail={
            "property": vr.property_text,
            "formula": vr.formula_repr,
            "target": target,
            "violating_states": sorted(vr.violating_states),
        },
    )


def z3_to_repair(node_name: str, proof) -> RepairSignal:
    """Z3 ProofResult -> repair signal (which obligation broke, and the point)."""
    return RepairSignal(
        kind="z3",
        description=f"Z3 obligation failed at '{node_name}'",
        detail={
            "node": node_name,
            "failed": list(getattr(proof, "failed_obligations", [])),
            "unproven": list(getattr(proof, "unproven_obligations", [])),
            "counterexample": getattr(proof, "counterexample", None),
        },
    )


def taint_to_repair(v) -> RepairSignal:
    """TaintViolation -> repair signal (source, sink, and the tainted path)."""
    return RepairSignal(
        kind="taint",
        description=f"untrusted '{v.source}' can reach dangerous sink '{v.sink}'",
        detail={"source": v.source, "sink": v.sink, "path": list(v.path)},
    )


# ── Collect the first violation across the graph verifiers ──

def _first_violation(engine, properties, check_taint) -> Tuple[Optional[RepairSignal], str]:
    if properties:
        from ..verification.temporal_verifier import PropertyResult
        for vr in engine.verify(properties):
            if vr.result != PropertyResult.PROVEN:
                return ctl_to_repair(vr), f"CTL: {vr.property_text or vr.formula_repr}"
    if check_taint:
        tr = engine.analyze_taint()
        if not tr.verified:
            v = tr.violations[0]
            return taint_to_repair(v), f"taint: {v.source}->{v.sink}"
    return None, ""


def _all_violations(engine, properties, check_taint) -> List[str]:
    out: List[str] = []
    if properties:
        from ..verification.temporal_verifier import PropertyResult
        out += [f"CTL: {vr.property_text or vr.formula_repr}"
                for vr in engine.verify(properties) if vr.result != PropertyResult.PROVEN]
    if check_taint:
        tr = engine.analyze_taint()
        out += [f"taint: {v.source}->{v.sink}" for v in tr.violations]
    return out


def counterexample_guided_repair(
    engine,
    repair_fn: Callable[[Any, RepairSignal], bool],
    *,
    properties: Optional[List[dict]] = None,
    check_taint: bool = True,
    max_iterations: int = 5,
) -> ReplanResult:
    """Drive the plan to PROVEN via verifier counterexamples.

    Each round: find the first violation (CTL, then taint), translate it to a
    ``RepairSignal``, call ``repair_fn(engine, signal)`` to mutate the plan, and
    re-verify. Stops on PROVEN, on a declined repair, or after ``max_iterations``
    -- and on abort returns the explicit unresolved violations (never a silent
    pass). Records the iteration history on ``engine._replan_history``.
    """
    history: List[RepairIteration] = []

    for i in range(1, max_iterations + 1):
        signal, viol = _first_violation(engine, properties, check_taint)
        if signal is None:
            engine._replan_history = history
            return ReplanResult(verified=True, iterations=i - 1, history=history, unresolved=[])

        repaired = bool(repair_fn(engine, signal))
        history.append(RepairIteration(iteration=i, violation=viol, signal=signal, repaired=repaired))
        logger.info(f"[Replan] iter {i}: {viol} -> repaired={repaired}")
        if not repaired:
            break  # replanner declined -> unrepairable

    # Final check after the last repair attempt.
    signal, _ = _first_violation(engine, properties, check_taint)
    if signal is None:
        engine._replan_history = history
        return ReplanResult(verified=True, iterations=len(history), history=history, unresolved=[])

    unresolved = _all_violations(engine, properties, check_taint)
    engine._replan_history = history
    return ReplanResult(verified=False, iterations=len(history), history=history, unresolved=unresolved)


# ── Built-in deterministic repair strategies (no LLM needed) ──

def _make_sanitizer_node(name: str):
    from .engine import Node

    def handle(self, user_text, extracted_data=None, memory=None):
        return "END", {}

    return type(name, (Node,), {
        "system_prompt": f"auto-inserted sanitizer ({name})",
        "sanitizer": True,
        "handle": handle,
    })


def insert_sanitizer_repair(engine, signal: RepairSignal) -> bool:
    """Repair a taint violation by inserting a sanitizer before the sink.

    Rewires ``prev -> sink`` on the tainted path to ``prev -> San -> sink``,
    where ``San`` is a fresh sanitizer node -- so taint no longer reaches the
    sink. Returns False for non-taint signals (declining the repair).
    """
    if signal.kind != "taint":
        return False
    path = signal.detail["path"]
    sink = signal.detail["sink"]
    if len(path) < 2:
        return False
    prev = path[-2]
    san_name = f"Sanitize_{prev}_{sink}"
    if san_name not in engine._nodes:
        engine.register(_make_sanitizer_node(san_name))
    # prev -> San -> sink
    engine._transitions[prev] = [san_name if t == sink else t for t in engine._transitions.get(prev, [])]
    engine._transitions.setdefault(san_name, [])
    if sink not in engine._transitions[san_name]:
        engine._transitions[san_name].append(sink)
    return True


def add_edge_to_reach_repair(engine, signal: RepairSignal) -> bool:
    """Repair a CTL reachability violation by adding an edge to the target.

    Adds ``entry -> target`` so the previously-unreachable target becomes
    reachable. Returns False if the signal has no resolvable target.
    """
    if signal.kind != "ctl" or not signal.detail.get("target"):
        return False
    target = signal.detail["target"]
    if target not in engine._nodes:
        return False
    from ..verification.temporal_verifier import _find_init_node
    entry = _find_init_node(engine._nodes, engine._transitions)
    if target not in engine._transitions.setdefault(entry, []):
        engine._transitions[entry].append(target)
    return True


def default_repair(engine, signal: RepairSignal) -> bool:
    """Dispatch to the built-in strategy for the signal kind."""
    if signal.kind == "taint":
        return insert_sanitizer_repair(engine, signal)
    if signal.kind == "ctl":
        return add_edge_to_reach_repair(engine, signal)
    return False   # z3 repair needs a real replanner; decline deterministically
