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
