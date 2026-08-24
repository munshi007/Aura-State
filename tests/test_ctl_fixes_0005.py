"""Regression tests for CTL/Kripke correctness bugs (task 0005).

These exercise the REAL pyModelChecking engine against known-answer graphs
(no mocks): a genuine dead-end, an unreachable terminal, and true/false
temporal ordering. See CLAUDE.md rule 8.
"""
from aura_state import (
    AuraEngine, Node, CompiledTransition,
    verify_engine, reachability, always_before, eventual_completion,
    PropertyResult,
)
from aura_state.verification.temporal_verifier import no_dead_ends


# ── Nodes (handle() bodies are irrelevant to structural/CTL verification) ──

def _node(name):
    return type(name, (Node,), {
        "system_prompt": f"{name} prompt",
        "handle": lambda self, user_text, extracted_data=None, memory=None: ("END", {}),
    })


Start = _node("Start")
Mid = _node("Mid")
End = _node("End")
Orphan = _node("Orphan")
BeforeNode = _node("BeforeNode")
AfterNode = _node("AfterNode")


# ═══════════════════════════════════════════════════════════════════
# A + B: structural dead-end detection, separate from totalization
# ═══════════════════════════════════════════════════════════════════

def test_ctl_dead_end_detected_fixes_0005():
    # Start -> Mid, where Mid is a sink that is NOT a declared terminal.
    engine = AuraEngine()
    engine.register(Start, Mid)
    engine.connect([CompiledTransition(from_node=Start, to_node=Mid)])

    # No node is declared an intended terminal -> Mid is an ACCIDENTAL dead-end.
    dead = no_dead_ends(engine._nodes, engine._transitions, terminals=set())
    assert dead.result == PropertyResult.VIOLATED
    assert "Mid" in dead.violating_states

    # Healthy graph: Start -> End, End IS the intended terminal -> no dead-ends.
    healthy = AuraEngine()
    healthy.register(Start, End)
    healthy.connect([CompiledTransition(from_node=Start, to_node=End)])
    ok = no_dead_ends(healthy._nodes, healthy._transitions, terminals={"End"})
    assert ok.result == PropertyResult.PROVEN
    assert ok.violating_states == set()


# ═══════════════════════════════════════════════════════════════════
# C: properties are judged at the INIT state, not over all states
# ═══════════════════════════════════════════════════════════════════

def test_ctl_init_state_fixes_0005():
    # Healthy: Start -> End (terminal). eventual_completion judged at init.
    engine = AuraEngine()
    engine.register(Start, End)
    engine.connect([CompiledTransition(from_node=Start, to_node=End)])
    results = verify_engine(engine, [
        {"description": "reaches End", "formula": eventual_completion("End")},
    ], terminals={"End"})
    assert results[0].result == PropertyResult.PROVEN

    # Unreachable terminal: Orphan is not connected to Start's reachable set.
    engine2 = AuraEngine()
    engine2.register(Start, Mid, Orphan)   # Start registered first -> init=Start
    engine2.connect([CompiledTransition(from_node=Start, to_node=Mid)])
    results2 = verify_engine(engine2, [
        {"description": "Orphan reachable", "formula": reachability("Orphan")},
    ], terminals={"Mid", "Orphan"})
    # Orphan is genuinely unreachable from the init state -> VIOLATED at init,
    # even though Orphan trivially satisfies EF(Orphan) at its own state.
    assert results2[0].result == PropertyResult.VIOLATED


# ═══════════════════════════════════════════════════════════════════
# D: always_before is real temporal ordering, not label co-occurrence
# ═══════════════════════════════════════════════════════════════════

def test_ctl_ordering_fixes_0005():
    # BeforeNode genuinely precedes AfterNode: Start -> BeforeNode -> AfterNode.
    good = AuraEngine()
    good.register(Start, BeforeNode, AfterNode)
    good.connect([
        CompiledTransition(from_node=Start, to_node=BeforeNode),
        CompiledTransition(from_node=BeforeNode, to_node=AfterNode),
    ])
    good_res = verify_engine(good, [
        {"description": "Before precedes After",
         "formula": always_before("BeforeNode", "AfterNode")},
    ], terminals={"AfterNode"})
    assert good_res[0].result == PropertyResult.PROVEN

    # AfterNode is reachable BEFORE BeforeNode ever holds: Start -> After -> Before.
    bad = AuraEngine()
    bad.register(Start, AfterNode, BeforeNode)
    bad.connect([
        CompiledTransition(from_node=Start, to_node=AfterNode),
        CompiledTransition(from_node=AfterNode, to_node=BeforeNode),
    ])
    bad_res = verify_engine(bad, [
        {"description": "Before precedes After",
         "formula": always_before("BeforeNode", "AfterNode")},
    ], terminals={"BeforeNode"})
    assert bad_res[0].result == PropertyResult.VIOLATED
