"""Regression tests for task 0007 bug sweep:
B1 cost recording, B2 cache-hit NameError, B3 ord embeddings,
B4 dead instructor fallback, B5 speculative extracted_data=None.
"""
import pytest
from pydantic import BaseModel

from aura_state.core.providers import LLMProvider
from aura_state.compiler.dspy_tuner import BootstrapTeleprompter, char_stub_embedder


# ── B1: CostTracker records real token usage ──

class _Usage:
    prompt_tokens = 1000
    completion_tokens = 500


class _Raw:
    usage = _Usage()


class _Dummy(BaseModel):
    ok: bool = True


class _Completions:
    def create_with_completion(self, **kwargs):
        return _Dummy(), _Raw()


class _Chat:
    completions = _Completions()


class _FakeClient:
    """Mocks only the LLM boundary (not the unit under test)."""
    chat = _Chat()


def test_cost_tracker_records_fixes_0007():
    provider = LLMProvider()
    provider.register_client("gpt", _FakeClient())
    provider.extract(
        model="gpt-4o-mini",
        response_model=_Dummy,
        messages=[{"role": "user", "content": "hi"}],
        node_name="N",
    )
    ct = provider.cost_tracker
    # gpt-4o-mini priced at 0.15/1M in, 0.60/1M out -> 1000 in + 500 out > $0.
    assert ct.total_cost_usd > 0
    report = ct.get_report()
    assert report["nodes"]["N"]["gpt-4o-mini"]["input_tokens"] == 1000
    assert report["nodes"]["N"]["gpt-4o-mini"]["output_tokens"] == 500


# (B2 obsolete: the GraphRAG trajectory cache was removed as feature-theater
#  during the design-time-verification refactor, so its logger bug is moot.)


# ── B3: default embedder is NOT the ord stub; stub only via injection ──

def test_embeddings_not_ord_fixes_0007():
    default = BootstrapTeleprompter()
    with pytest.raises(RuntimeError):
        default.compile([{"success": True, "node": "N", "input": "x", "output": {}}])

    injected = BootstrapTeleprompter(embedder=char_stub_embedder)
    injected.compile([{"success": True, "node": "N", "input": "hello world", "output": {"a": 1}}])
    out = injected.optimize_node("N", "PROMPT", "hello world")
    assert "FEW-SHOT" in out


# ── B4: extraction always routes through the provider (no dead client branch) ──

def test_extraction_uses_provider_not_dead_branch_fixes_0007(monkeypatch):
    from aura_state.core.engine import AuraEngine, Node, CompiledTransition

    class Lead(BaseModel):
        name: str = "x"

    class A(Node):
        system_prompt = "extract"
        extracts = Lead

        def handle(self, user_text, extracted_data=None, memory=None):
            return "END", {}

    engine = AuraEngine()  # no real OpenAI client (instructor would reject a fake)
    engine.register(A)
    # Simulate a live client + registered provider without instructor wrapping.
    engine.client = object()
    engine.provider.register_client("gpt", _FakeClient())

    called = {"n": 0}

    def spy(**kwargs):
        called["n"] += 1
        return Lead()

    monkeypatch.setattr(engine.provider, "extract", spy)
    # Extraction (stage 3) runs before routing; A has no outgoing edge so the
    # call dead-ends afterwards -- irrelevant to what we assert here.
    from aura_state.core.exceptions import StateTransitionError
    try:
        engine.process("A", "some text")
    except StateTransitionError:
        pass
    assert called["n"] >= 1  # provider path taken; dead self.client branch gone


# (B5 obsolete: speculative execution was removed as feature-theater during the
#  design-time-verification refactor -- no handler runs on fabricated data now.)
