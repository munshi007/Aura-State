"""
Tests for core innovations.
Tests all 5 innovations without requiring live LLM API keys.
"""
import pytest
import time
from unittest.mock import MagicMock, patch
from pydantic import BaseModel

from aura_state.core.engine import AuraEngine, Node, CompiledTransition
from aura_state.core.adaptive_graph import AdaptiveDAG, NodeHealthMetrics, RuntimeEdge
from aura_state.core.verification_loop import VerificationLoop, ReflectionMemory, Reflection
from aura_state.core.providers import LLMProvider, CostTracker, MODEL_PRICING
from aura_state.compiler.schema_compiler import (
    levenshtein_distance, suggest_field, compile_pydantic_model, compile_schema
)


# ═══════════════════════════════════════════════════════════════
# Speculative Node Execution
# ═══════════════════════════════════════════════════════════════

class GreetNode(Node):
    system_prompt = "Greet the user."
    def handle(self, user_text, extracted_data=None, memory=None):
        return "QualifyNode", {"greeting": "hello"}

class QualifyNode(Node):
    system_prompt = "Qualify the lead."
    def handle(self, user_text, extracted_data=None, memory=None):
        return "EndNode", {"qualified": True}

class EndNode(Node):
    system_prompt = "End the conversation."
    def handle(self, user_text, extracted_data=None, memory=None):
        return "END", {}


class TestSpeculativeExecution:
    def test_engine_has_speculation_cache(self):
        engine = AuraEngine(speculation_depth=1)
        assert hasattr(engine, '_pending_speculations')
        assert hasattr(engine, '_speculation_depth')
        assert engine._speculation_depth == 1
    
    def test_speculation_depth_configurable(self):
        engine = AuraEngine(speculation_depth=3)
        assert engine._speculation_depth == 3
    
    def test_check_speculation_miss(self):
        engine = AuraEngine()
        result = engine._check_speculation("NonExistent")
        assert result is None


# ═══════════════════════════════════════════════════════════════
# Adaptive Compute Graph
# ═══════════════════════════════════════════════════════════════

class TestAdaptiveDAG:
    def test_record_execution(self):
        dag = AdaptiveDAG()
        dag.record_execution("NodeA", success=True, latency_ms=50.0)
        health = dag.get_health("NodeA")
        assert health.total_executions == 1
        assert health.failures == 0
        assert health.avg_latency_ms == 50.0
    
    def test_fail_rate(self):
        dag = AdaptiveDAG()
        dag.record_execution("NodeA", success=True, latency_ms=10)
        dag.record_execution("NodeA", success=False, latency_ms=10)
        assert dag.get_health("NodeA").fail_rate == 0.5
    
    def test_reflexion_injection_threshold(self):
        dag = AdaptiveDAG()
        # Not triggered yet
        assert dag.should_inject_reflexion("NodeA") is False
        
        # 3 consecutive failures triggers reflexion
        for _ in range(3):
            dag.record_execution("NodeA", success=False, latency_ms=10)
        assert dag.should_inject_reflexion("NodeA") is True
        
        # After marking, doesn't trigger again
        dag.mark_reflexion_injected("NodeA")
        assert dag.should_inject_reflexion("NodeA") is False
    
    def test_llm_bypass_threshold(self):
        dag = AdaptiveDAG()
        # Need MIN_EXECUTIONS_FOR_BYPASS first
        for _ in range(10):
            dag.record_execution("NodeA", success=True, latency_ms=5, cache_hit=True)
        assert dag.should_bypass_llm("NodeA") is True
    
    def test_edge_proposal(self):
        dag = AdaptiveDAG()
        dag.propose_edge("NodeA", "NodeC", confidence=0.85, evidence="MCTS simulation #4")
        assert dag.total_proposed_edges == 1
        
        proposals = dag.get_proposed_edges()
        assert len(proposals) == 1
        assert proposals[0].from_node == "NodeA"
        assert proposals[0].confidence == 0.85
    
    def test_accept_edge(self):
        dag = AdaptiveDAG()
        dag.propose_edge("NodeA", "NodeC", confidence=0.9, evidence="test")
        assert dag.accept_edge("NodeA", "NodeC") is True
        assert len(dag.get_proposed_edges()) == 0  # No unaccepted proposals
    
    def test_health_report(self):
        dag = AdaptiveDAG()
        dag.record_execution("NodeA", success=True, latency_ms=50)
        dag.record_execution("NodeB", success=False, latency_ms=100)
        report = dag.get_health_report()
        assert "NodeA" in report
        assert "NodeB" in report
        assert report["NodeA"]["fail_rate"] == 0.0
        assert report["NodeB"]["fail_rate"] == 1.0


# ═══════════════════════════════════════════════════════════════
# Verification Loop
# ═══════════════════════════════════════════════════════════════

class TestVerificationLoop:
    def test_reflexion_memory_add_and_retrieve(self):
        mem = ReflectionMemory()
        r = Reflection(
            node_name="Test", input_text="hello", extracted_data={"x": 1},
            error="wrong value", critique="should be 2", attempt=1
        )
        mem.add(r)
        assert mem.total_reflections == 1
        assert len(mem.get_reflections("Test")) == 1
    
    def test_reflexion_memory_ring_buffer(self):
        mem = ReflectionMemory(max_reflections_per_node=3)
        for i in range(5):
            mem.add(Reflection(
                node_name="Test", input_text=f"input_{i}",
                extracted_data={"i": i}, error="err", critique="fix", attempt=i
            ))
        # Ring buffer should keep only last 3
        assert len(mem.get_reflections("Test")) == 3
        assert mem.get_reflections("Test")[0].attempt == 2  # oldest kept
    
    def test_format_negative_examples(self):
        mem = ReflectionMemory()
        mem.add(Reflection(
            node_name="Test", input_text="hello",
            extracted_data={"x": 1}, error="wrong", critique="fix it", attempt=1
        ))
        text = mem.format_as_negative_examples("Test")
        assert "Previous Failures" in text
        assert "wrong" in text
    
    def test_verification_passes_without_sandbox_rule(self):
        vl = VerificationLoop()
        passed, error = vl.verify_extraction("N", None, None, None, "text")
        assert passed is True
    
    def test_verification_loop_runs(self):
        vl = VerificationLoop(max_iterations=2)
        mock_model = MagicMock()
        mock_model.model_dump.return_value = {"value": 42}
        
        def fake_extract(prompt, text):
            return mock_model
        
        result, iterations, verified = vl.run(
            node_name="Test", user_text="test",
            system_prompt="extract", extract_fn=fake_extract,
            sandbox_rule=None, sandbox=None,
        )
        assert verified is True
        assert iterations == 1


# ═══════════════════════════════════════════════════════════════
# Schema-Driven Node Compilation
# ═══════════════════════════════════════════════════════════════

class TestSchemaCompiler:
    def test_levenshtein_distance_identical(self):
        assert levenshtein_distance("hello", "hello") == 0
    
    def test_levenshtein_distance_one_edit(self):
        assert levenshtein_distance("cat", "car") == 1
    
    def test_levenshtein_distance_empty(self):
        assert levenshtein_distance("", "abc") == 3
        assert levenshtein_distance("abc", "") == 3
    
    def test_suggest_field_exact_match(self):
        fields = ["wall_area", "cost", "quantity"]
        assert suggest_field("wall_area", fields) == "wall_area"
    
    def test_suggest_field_fuzzy(self):
        fields = ["wall_area", "cost", "quantity"]
        assert suggest_field("wall_aera", fields) == "wall_area"  # 1 edit
    
    def test_suggest_field_too_far(self):
        fields = ["wall_area"]
        assert suggest_field("completely_different_name", fields) is None
    
    def test_compile_pydantic_model(self):
        schema = {
            "properties": {
                "name": {"type": "string", "description": "User name"},
                "age": {"type": "integer", "description": "User age", "minimum": 0, "maximum": 150},
            },
            "required": ["name"]
        }
        Model = compile_pydantic_model("TestModel", schema)
        
        instance = Model(name="Alice", age=30)
        assert instance.name == "Alice"
        assert instance.age == 30
    
    def test_compile_pydantic_model_optional_fields(self):
        schema = {
            "properties": {
                "name": {"type": "string", "description": "Name"},
                "nickname": {"type": "string", "description": "Optional nickname"},
            },
            "required": ["name"]
        }
        Model = compile_pydantic_model("TestModel2", schema)
        instance = Model(name="Bob")
        assert instance.name == "Bob"
        assert instance.nickname is None
    
    def test_compile_schema_to_node(self):
        schema = {
            "title": "RoomData",
            "description": "Extract room dimensions",
            "properties": {
                "wall_area_sqft": {"type": "number", "description": "Total wall area in sqft"},
            },
            "required": ["wall_area_sqft"]
        }
        NodeClass = compile_schema(schema)
        assert NodeClass.__name__ == "RoomData"
        assert NodeClass.system_prompt != ""
        assert NodeClass.extracts is not None


# ═══════════════════════════════════════════════════════════════
# Multi-Provider LLM Orchestration
# ═══════════════════════════════════════════════════════════════

class TestCostTracker:
    def test_record_and_report(self):
        ct = CostTracker()
        ct.record("NodeA", "gpt-4o", input_tokens=1000, output_tokens=500, latency_ms=200)
        report = ct.get_report()
        assert report["nodes"]["NodeA"]["gpt-4o"]["calls"] == 1
    
    def test_budget_enforcement(self):
        ct = CostTracker(budget_usd=0.001)
        ct.record("NodeA", "gpt-4.5-preview", input_tokens=1_000_000, output_tokens=0, latency_ms=100)
        assert ct.is_over_budget() is True
    
    def test_no_budget_never_over(self):
        ct = CostTracker()
        ct.record("NodeA", "gpt-4o", input_tokens=1_000_000, output_tokens=1_000_000, latency_ms=100)
        assert ct.is_over_budget() is False


class TestLLMProvider:
    def test_register_client(self):
        provider = LLMProvider()
        mock_client = MagicMock()
        provider.register_client("gpt", mock_client)
        assert "gpt" in provider._clients
    
    def test_failover_chain(self):
        provider = LLMProvider()
        provider.register_client("gpt", MagicMock())
        provider.register_client("claude", MagicMock())
        provider.set_failover_chain(["gpt", "claude"])
        
        failover = provider._get_failover_model("gpt-4o")
        assert failover == "claude-3.5-sonnet"
    
    def test_failover_chain_end(self):
        provider = LLMProvider()
        provider.register_client("claude", MagicMock())
        provider.set_failover_chain(["claude"])
        
        failover = provider._get_failover_model("claude-3.5-sonnet")
        assert failover is None  # No more fallbacks


# ═══════════════════════════════════════════════════════════════
# Integration: Engine has all innovations
# ═══════════════════════════════════════════════════════════════

class TestEngineInnovations:
    def test_engine_has_adaptive_graph(self):
        engine = AuraEngine()
        assert isinstance(engine.adaptive_graph, AdaptiveDAG)
    
    def test_engine_has_verification_loop(self):
        engine = AuraEngine()
        assert isinstance(engine.verification_loop, VerificationLoop)
    
    def test_engine_has_provider(self):
        engine = AuraEngine()
        assert isinstance(engine.provider, LLMProvider)
    
    def test_node_has_model_field(self):
        class TestNode(Node):
            system_prompt = "Test"
            model = "claude-3.5-sonnet"
        
        assert TestNode.model == "claude-3.5-sonnet"
    
    def test_health_report_empty(self):
        engine = AuraEngine()
        report = engine.health_report()
        assert isinstance(report, dict)
    
    def test_cost_report_empty(self):
        engine = AuraEngine()
        report = engine.cost_report()
        assert isinstance(report, dict)
        assert "total_cost_usd" in report
    
    def test_budget_configurable(self):
        engine = AuraEngine(budget_usd=10.0)
        assert engine.provider.cost_tracker._budget_usd == 10.0
