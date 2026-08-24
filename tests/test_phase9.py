"""Tests for formal verification innovations: temporal verification, conformal prediction, proof engine."""
import pytest
from pydantic import BaseModel
from aura_state import (
    AuraEngine, Node, CompiledTransition,
    compile_kripke, verify_engine, verify_property,
    reachability, always_before, mutual_exclusion, eventual_completion,
    PropertyResult,
    conformal_interval, conformal_from_extractions,
    prove_extraction, prove_consistency,
)


# ── Shared test nodes ──

class IntakeNode(Node):
    system_prompt = "Gather user information."
    def handle(self, user_text, extracted_data=None, memory=None):
        return "ReviewNode", {"status": "gathered"}

class ReviewNode(Node):
    system_prompt = "Review the gathered data."
    def handle(self, user_text, extracted_data=None, memory=None):
        return "ApproveNode", {"status": "reviewed"}

class ApproveNode(Node):
    system_prompt = "Approve the request."
    def handle(self, user_text, extracted_data=None, memory=None):
        return "END", {"status": "approved"}

class RejectNode(Node):
    system_prompt = "Reject the request."
    def handle(self, user_text, extracted_data=None, memory=None):
        return "END", {"status": "rejected"}


def _build_engine():
    engine = AuraEngine()
    engine.register(IntakeNode, ReviewNode, ApproveNode, RejectNode)
    engine.connect([
        CompiledTransition(from_node=IntakeNode, to_node=ReviewNode),
        CompiledTransition(from_node=ReviewNode, to_node=ApproveNode),
        CompiledTransition(from_node=ReviewNode, to_node=RejectNode),
    ])
    return engine


# ═══════════════════════════════════════════════
# Temporal Logic Verification
# ═══════════════════════════════════════════════

class TestTemporalVerifier:
    def test_kripke_compilation(self):
        engine = _build_engine()
        kripke = compile_kripke(engine._nodes, engine._transitions)
        states = kripke.states()
        assert "IntakeNode" in states
        assert "ReviewNode" in states
        assert "ApproveNode" in states
        assert "RejectNode" in states

    def test_reachability_proven(self):
        engine = _build_engine()
        results = verify_engine(engine, [
            {"description": "ReviewNode is reachable", "formula": reachability("ReviewNode")}
        ])
        assert "IntakeNode" in results[0].satisfying_states
        assert "ReviewNode" in results[0].satisfying_states

    def test_mutual_exclusion(self):
        engine = _build_engine()
        results = verify_engine(engine, [
            {"description": "Cannot be Approve and Reject", "formula": mutual_exclusion("ApproveNode", "RejectNode")}
        ])
        assert results[0].result == PropertyResult.PROVEN

    def test_always_before(self):
        engine = _build_engine()
        formula = always_before("ReviewNode", "ApproveNode")
        results = verify_engine(engine, [
            {"description": "Review before Approve", "formula": formula}
        ])
        assert len(results) == 1

    def test_verify_engine_returns_list(self):
        engine = _build_engine()
        results = verify_engine(engine, [
            {"description": "Reachability", "formula": reachability("ApproveNode")},
            {"description": "Exclusion", "formula": mutual_exclusion("ApproveNode", "RejectNode")},
        ])
        assert len(results) == 2

    def test_terminal_nodes_self_loop(self):
        engine = _build_engine()
        kripke = compile_kripke(engine._nodes, engine._transitions)
        transitions = kripke.transitions()
        terminal_loops = [(a, b) for a, b in transitions if a == b]
        assert len(terminal_loops) >= 2


# ═══════════════════════════════════════════════
# Conformal Prediction
# ═══════════════════════════════════════════════

class TestConformalPrediction:
    def test_single_value(self):
        iv = conformal_interval([100.0])
        assert iv.point_estimate == 100.0
        assert iv.n_samples == 1

    def test_two_values(self):
        iv = conformal_interval([90.0, 110.0])
        assert iv.lower == 90.0
        assert iv.upper == 110.0
        assert iv.n_samples == 2

    def test_tight_values(self):
        iv = conformal_interval([100.0, 100.0, 100.0, 100.0, 100.0])
        assert iv.lower == iv.upper == 100.0

    def test_spread_values(self):
        iv = conformal_interval([90.0, 95.0, 100.0, 105.0, 110.0], confidence=0.95)
        assert iv.lower < 100.0
        assert iv.upper > 100.0
        assert iv.point_estimate == 100.0

    def test_conformal_from_extractions(self):
        class MockExtraction(BaseModel):
            cost: float
            area: float

        extractions = [
            MockExtraction(cost=1000.0, area=500.0),
            MockExtraction(cost=1050.0, area=510.0),
            MockExtraction(cost=980.0, area=490.0),
            MockExtraction(cost=1020.0, area=505.0),
        ]

        result = conformal_from_extractions(extractions, confidence=0.95)
        assert "cost" in result.intervals
        assert "area" in result.intervals
        assert result.coverage_level == 0.95

    def test_empty_extractions(self):
        result = conformal_from_extractions([])
        assert not result.calibrated
        assert result.intervals == {}


# ═══════════════════════════════════════════════
# Z3 Proof Engine
# ═══════════════════════════════════════════════

class TestProofEngine:
    def test_valid_extraction(self):
        data = {"area": 500, "cost_per_sqft": 3, "total_cost": 1500}
        result = prove_extraction(data, ["total_cost == area * cost_per_sqft"])
        assert result.verified

    def test_invalid_extraction(self):
        data = {"area": 500, "cost_per_sqft": 3, "total_cost": 9999}
        result = prove_extraction(data, ["total_cost == area * cost_per_sqft"])
        assert not result.verified
        assert "total_cost == area * cost_per_sqft" in result.failed_obligations

    def test_range_constraint(self):
        data = {"price": 50000}
        result = prove_extraction(data, ["price > 0", "price < 1000000"])
        assert result.verified

    def test_negative_price(self):
        data = {"price": -100}
        result = prove_extraction(data, ["price > 0"])
        assert not result.verified

    def test_no_obligations(self):
        result = prove_extraction({"x": 1}, [])
        assert result.verified

    def test_consistency_alias(self):
        data = {"unit": 10, "qty": 5, "total": 50}
        result = prove_consistency(data, ["total == unit * qty"])
        assert result.verified

    def test_counterexample_has_details(self):
        data = {"margin": -5}
        result = prove_extraction(data, ["margin >= 0"])
        assert not result.verified
        assert result.counterexample is not None
        assert "failed_constraints" in result.counterexample


# ── Regression: proof engine must fail CLOSED, no eval RCE (task 0002) ──

class TestProofEngineFailsClosed:
    """The proof engine previously failed OPEN: an obligation it could not
    parse was silently skipped and counted as passed, and obligations were
    evaluated with eval(). Both are load-bearing correctness bugs."""

    def test_unparseable_obligation_is_unproven_not_passed_fixes_0002(self):
        # Garbage that cannot compile must NOT be reported as verified.
        result = prove_extraction({"x": 10}, ["x >>> 0 !!"])
        assert not result.verified
        assert "x >>> 0 !!" in result.unproven_obligations
        assert "x >>> 0 !!" not in result.failed_obligations

    def test_unknown_variable_is_unproven_not_passed_fixes_0002(self):
        # Obligation references a field that was never extracted.
        result = prove_extraction({"x": 10}, ["y > 0"])
        assert not result.verified
        assert "y > 0" in result.unproven_obligations

    def test_obligation_over_string_field_is_unproven_fixes_0002(self):
        # Non-numeric data cannot back a numeric obligation -> unproven, not passed.
        result = prove_extraction({"name": "sarah"}, ["name > 0"])
        assert not result.verified
        assert "name > 0" in result.unproven_obligations

    def test_rce_subclasses_gadget_is_rejected_fixes_0002(self):
        # The classic {"__builtins__": {}} eval escape. Must not execute; the
        # obligation must be rejected as unproven, and verified stays False.
        gadget = "().__class__.__bases__[0].__subclasses__()"
        result = prove_extraction({"x": 1}, [gadget])
        assert not result.verified
        assert gadget in result.unproven_obligations

    def test_valid_obligation_still_proves_true(self):
        result = prove_extraction({"area": 100, "rate": 3, "total": 300},
                                  ["total == area * rate", "area > 0"])
        assert result.verified
        assert result.unproven_obligations == []

    def test_violated_obligation_reports_counterexample(self):
        result = prove_extraction({"area": 100, "rate": 3, "total": 999},
                                  ["total == area * rate"])
        assert not result.verified
        assert "total == area * rate" in result.failed_obligations

    def test_chained_comparison_supported(self):
        assert prove_extraction({"p": 5}, ["0 < p < 10"]).verified
        assert not prove_extraction({"p": 50}, ["0 < p < 10"]).verified

    def test_mixed_valid_and_unproven_fails_closed(self):
        # One provable, one garbage -> overall must be False.
        result = prove_extraction({"x": 10}, ["x > 0", "bogus $%"])
        assert not result.verified
        assert result.failed_obligations == []
        assert "bogus $%" in result.unproven_obligations
