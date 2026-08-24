"""The headline contract: verification runs INSIDE process(), not beside it.

Before the design-time-verification refactor, prove_extraction / conformal were
only reachable from user code and tests; process() ran the sandbox rule only.
These tests pin that Z3 obligations and conformal intervals now execute in the
loop, and that a decision node's rule fires even with no LLM extraction.
"""
from pydantic import BaseModel

from aura_state.core.engine import AuraEngine, Node, CompiledTransition


class Quote(BaseModel):
    area: int = 100
    rate: int = 3
    total: int = 300


class _Chat:
    def __init__(self, obj):
        self._obj = obj
        self.completions = self

    def create_with_completion(self, **kwargs):
        return self._obj, type("R", (), {"usage": None})()


class _Client:
    def __init__(self, obj):
        self.chat = _Chat(obj)


def _engine_with_extraction(returned: BaseModel):
    """Engine whose provider yields a fixed extraction (LLM boundary mocked)."""
    e = AuraEngine()
    e.client = object()
    e.provider.register_client("gpt", _Client(returned))
    return e


def test_z3_obligation_runs_in_process_and_passes():
    class Priced(Node):
        system_prompt = "price it"
        extracts = Quote
        obligations = ["total == area * rate", "area > 0"]

        def handle(self, user_text, extracted_data=None, memory=None):
            return "END", {}

    e = _engine_with_extraction(Quote(area=100, rate=3, total=300))
    e.register(Priced)
    e._transitions["Priced"] = ["END"]
    e.process("Priced", "quote please")
    rep = e.verification_reports()[-1]
    assert rep["extraction_verified"] is True


def test_z3_obligation_failure_is_caught_in_loop():
    class Priced(Node):
        system_prompt = "price it"
        extracts = Quote
        obligations = ["total == area * rate"]  # 100*3 != 999

        def handle(self, user_text, extracted_data=None, memory=None):
            return "END", {}

    e = _engine_with_extraction(Quote(area=100, rate=3, total=999))
    e.register(Priced)
    e._transitions["Priced"] = ["END"]
    e.process("Priced", "quote please")
    rep = e.verification_reports()[-1]
    # The bad extraction cannot be verified -> the loop reports it (fail-closed).
    assert rep["extraction_verified"] is False


def test_conformal_interval_produced_over_consensus_runs():
    class Priced(Node):
        system_prompt = "price it"
        extracts = Quote
        consensus = 3
        confidence = 0.9

        def handle(self, user_text, extracted_data=None, memory=None):
            return "END", {}

    e = _engine_with_extraction(Quote(area=100, rate=3, total=300))
    e.register(Priced)
    e._transitions["Priced"] = ["END"]
    e.process("Priced", "quote please")
    rep = e.verification_reports()[-1]
    assert "conformal" in rep  # a ConformalResult was computed in the loop


def test_decision_node_rule_fires_without_extraction():
    # Flagship case: a node with a sandbox_rule and NO extracts. Previously the
    # rule never ran because it was gated behind `if node.extracts`.
    class Qualify(Node):
        system_prompt = "qualify"
        sandbox_rule = "result = budget > 100000"

        def handle(self, user_text, extracted_data=None, memory=None):
            return "END", {}

    e = AuraEngine()
    e.register(Qualify)
    e._transitions["Qualify"] = ["END"]
    e.process("Qualify", "text", memory={"budget": 450000})
    rep = e.verification_reports()[-1]
    assert rep["contract_verified"] is True

    e.process("Qualify", "text", memory={"budget": 50000})
    assert e.verification_reports()[-1]["contract_verified"] is False
