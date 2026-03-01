"""
Tests for AuraEngine.
"""
import pytest
from typing import Dict, Any, Optional
from pydantic import BaseModel
from aura_state import AuraEngine, Node, CompiledTransition, StateTransitionError, ConsensusStrategy
from aura_state.consensus.auto_vote import AutoConsensus
from aura_state.memory.pruner import ContextPruner


# ═══════════════════════════════════════════════
# Mock Nodes
# ═══════════════════════════════════════════════

class SimpleExtraction(BaseModel):
    value: int
    intent: str


class StartNode(Node):
    system_prompt = "Welcome the user."
    
    def handle(self, user_text: str, extracted_data=None, memory=None):
        return "EndNode", {"status": "success"}


class EndNode(Node):
    system_prompt = "Say goodbye."
    
    def handle(self, user_text: str, extracted_data=None, memory=None):
        return "DONE", {}


class DeadEndNode(Node):
    system_prompt = "This node has no outgoing edges."
    
    def handle(self, user_text: str, extracted_data=None, memory=None):
        return "NOWHERE", {}


# ═══════════════════════════════════════════════
# Test: Engine Registration & Graph Traversal
# ═══════════════════════════════════════════════

def test_engine_registration_and_traversal():
    engine = AuraEngine()
    engine.register(StartNode, EndNode)
    engine.connect([
        CompiledTransition(from_node=StartNode, to_node=EndNode)
    ])
    
    next_state, payload = engine.process("StartNode", "Hello")
    assert next_state == "EndNode"
    assert payload["status"] == "success"


def test_missing_node():
    engine = AuraEngine()
    with pytest.raises(StateTransitionError, match="not registered"):
        engine.process("NonExistentNode", "Test")


def test_dead_end_raises():
    engine = AuraEngine()
    engine.register(DeadEndNode)
    
    with pytest.raises(StateTransitionError, match="dead end"):
        engine.process("DeadEndNode", "Test")


# ═══════════════════════════════════════════════
# Test: Core internals are always active
# ═══════════════════════════════════════════════

def test_tracer_is_always_active():
    engine = AuraEngine()
    assert engine.tracer is not None, "AuraTrace must always be initialized"


def test_cache_is_always_active():
    engine = AuraEngine()
    assert engine.cache is not None, "GraphRAGCache must always be initialized"


def test_compiler_is_always_active():
    engine = AuraEngine()
    assert engine.compiler is not None, "BootstrapTeleprompter must always be initialized"


# ═══════════════════════════════════════════════
# Test: Context Pruner (utility)
# ═══════════════════════════════════════════════

def test_context_pruner():
    history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "What's my balance?"},
        {"role": "assistant", "content": "$100"}
    ]
    
    pruned = ContextPruner.prune(history, max_messages=2)
    assert len(pruned) == 3  # System + 2 messages
    assert pruned[0]["role"] == "system"
    assert pruned[-1]["content"] == "$100"
    
    pruned_req = ContextPruner.prune(history, required_keys=["account_id", "user_tier"], max_messages=2)
    assert len(pruned_req) == 4
    assert "account_id" in pruned_req[1]["content"]


# ═══════════════════════════════════════════════
# Test: Auto-Consensus
# ═══════════════════════════════════════════════

def test_auto_consensus_majority():
    runs = [
        SimpleExtraction(value=10, intent="buy"),
        SimpleExtraction(value=10, intent="buy"),
        SimpleExtraction(value=5, intent="sell"),
        SimpleExtraction(value=10, intent="buy"),
    ]
    
    result = AutoConsensus.resolve(runs, strategy=ConsensusStrategy.MAJORITY_VOTE)
    assert result.value == 10
    assert result.intent == "buy"

def test_auto_consensus_unanimous():
    runs = [
        SimpleExtraction(value=10, intent="buy"),
        SimpleExtraction(value=10, intent="buy"),
        SimpleExtraction(value=5, intent="sell"),
    ]
    
    with pytest.raises(ValueError, match="Unanimous consensus required"):
        AutoConsensus.resolve(runs, strategy=ConsensusStrategy.UNANIMOUS)

    unanimous_runs = [
        SimpleExtraction(value=10, intent="buy"),
        SimpleExtraction(value=10, intent="buy"),
    ]
    result = AutoConsensus.resolve(unanimous_runs, strategy=ConsensusStrategy.UNANIMOUS)
    assert result.value == 10
