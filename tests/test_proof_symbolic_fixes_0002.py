"""Task 0002-C: symbolic consistency proof — obligations range freely over
declared bounds (not pinned), so a self-contradictory spec is caught before
deploy. Real Z3, no mocks."""
from pydantic import BaseModel, Field

from aura_state import (
    prove_obligations_satisfiable, field_bounds_from_model,
    AuraEngine, Node, CompiledTransition,
)


def test_consistent_obligations_satisfiable_fixes_0002():
    r = prove_obligations_satisfiable(["total == area * rate", "area > 0", "rate > 0"])
    assert r.satisfiable is True
    assert r.witness is not None
    # The witness genuinely satisfies the relationship (up to float rounding).
    w = r.witness
    assert abs(w["total"] - w["area"] * w["rate"]) < 1e-6
    assert w["area"] > 0 and w["rate"] > 0


def test_contradictory_obligations_unsat_fixes_0002():
    # No value of x can satisfy both -> the spec is impossible, caught symbolically.
    r = prove_obligations_satisfiable(["x > 5", "x < 3"])
    assert r.satisfiable is False
    assert r.witness is None
    assert "unsatisfiable" in (r.reason or "")


def test_bounds_make_obligation_impossible_fixes_0002():
    # x < 0 is satisfiable on its own...
    assert prove_obligations_satisfiable(["x < 0"]).satisfiable is True
    # ...but not once the schema pins x >= 0.
    r = prove_obligations_satisfiable(["x < 0"], bounds={"x": {"ge": 0}})
    assert r.satisfiable is False


def test_field_bounds_from_pydantic_fixes_0002():
    class Q(BaseModel):
        area: int = Field(ge=0)
        rate: int = Field(gt=0, le=10)

    b = field_bounds_from_model(Q)
    assert b["area"] == {"ge": 0}
    assert b["rate"] == {"gt": 0, "le": 10}


def test_contract_flags_inconsistent_obligations_fixes_0002():
    class Bad(Node):
        system_prompt = "impossible spec"
        obligations = ["x > 5", "x < 3"]

        def handle(self, user_text, extracted_data=None, memory=None):
            return "END", {}

    class Good(Node):
        system_prompt = "fine spec"
        obligations = ["a >= 0"]

        def handle(self, user_text, extracted_data=None, memory=None):
            return "END", {}

    e = AuraEngine()
    e.register(Bad, Good)
    e.connect([CompiledTransition(from_node=Bad, to_node=Good)])
    c = e.compile_contract()
    bad = next(n for n in c.nodes if n.name == "Bad")
    good = next(n for n in c.nodes if n.name == "Good")
    assert bad.obligations_consistent is False    # self-contradictory -> flagged
    assert good.obligations_consistent is True


def test_boolean_obligations_symbolic_and_pointwise():
    """Boolean obligations (e.g. read_only == True) must compile — surfaced by
    modelling the LangGraph SQL agent, whose GenSQL node proves read_only.

    Before this fix prove_obligations_satisfiable declared every var as Real, so
    `read_only == True` crashed Z3 with a parser error.
    """
    from aura_state.verification.proof_engine import (
        prove_obligations_satisfiable, prove_extraction,
    )
    # symbolic SAT: a bool obligation is satisfiable; its negation-pair is not
    assert prove_obligations_satisfiable(["read_only == True"]).satisfiable is True
    assert prove_obligations_satisfiable(
        ["read_only == True", "read_only == False"]
    ).satisfiable is False
    # bool + numeric mixed in one set still solves
    assert prove_obligations_satisfiable(
        ["amount >= 0", "amount <= 500", "read_only == True"]
    ).satisfiable is True
    # point check fails CLOSED on the adversarial (read_only False) input
    assert prove_extraction({"read_only": True}, ["read_only == True"]).verified is True
    assert prove_extraction({"read_only": False}, ["read_only == True"]).verified is False
