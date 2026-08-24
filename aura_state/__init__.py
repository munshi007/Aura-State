"""
Aura-State: LLM State Machine Compiler.

Usage:
    from aura_state import AuraEngine, Node, CompiledTransition
"""
from .core.engine import AuraEngine, Node, CompiledTransition
from .execution.tracer import AuraTrace
from .compiler.dspy_tuner import BootstrapTeleprompter
from .execution.sandbox import SandboxedInterpreter, SandboxExecutionError
from .memory.pruner import ContextPruner
from .consensus.auto_vote import ConsensusStrategy
from .core.exceptions import AuraStateError, StateTransitionError, MaxRetriesExceededError
from .loaders.json_graph import JSONGraphLoader

# ── Core Innovations ──
from .core.adaptive_graph import AdaptiveDAG, NodeHealthMetrics, EdgeStats
from .core.verification_loop import VerificationLoop, ReflectionMemory, Reflection
from .core.providers import LLMProvider, CostTracker
from .compiler.schema_compiler import compile_schema, compile_openapi_schemas, levenshtein_distance

from .verification.temporal_verifier import (
    compile_kripke, verify_engine, verify_property,
    reachability, always_before, mutual_exclusion, eventual_completion,
    PropertyResult, VerificationResult,
)
from .verification.conformal import (
    conformal_interval, conformal_from_extractions,
    PredictionInterval, ConformalResult,
)
from .verification.proof_engine import prove_extraction, prove_consistency, ProofResult
from .compiler.spec_compiler import (
    compile_contract, diff_contracts, check_faithfulness,
    AuraContract, NodeContract, PropertyVerdict, TaintContract, TaintPath, ContractError,
)
from .verification.taint import analyze_taint, TaintResult, TaintViolation
from .verification.risk_control import RiskController, learn_then_test

__all__ = [
    "AuraEngine",
    "Node",
    "CompiledTransition",
    "AuraTrace",
    "BootstrapTeleprompter",
    "AdaptiveDAG",
    "NodeHealthMetrics",
    "EdgeStats",
    "VerificationLoop",
    "ReflectionMemory",
    "Reflection",
    "LLMProvider",
    "CostTracker",
    "compile_schema",
    "compile_openapi_schemas",
    "levenshtein_distance",
    "ContextPruner",
    "ConsensusStrategy",
    "SandboxedInterpreter",
    "SandboxExecutionError",
    "JSONGraphLoader",
    "AuraStateError",
    "StateTransitionError",
    "MaxRetriesExceededError",
    # Formal Verification
    "compile_kripke",
    "verify_engine",
    "verify_property",
    "reachability",
    "always_before",
    "mutual_exclusion",
    "eventual_completion",
    "PropertyResult",
    "VerificationResult",
    "conformal_interval",
    "conformal_from_extractions",
    "PredictionInterval",
    "ConformalResult",
    "prove_extraction",
    "prove_consistency",
    "ProofResult",
    # Design→spec compiler
    "compile_contract",
    "diff_contracts",
    "check_faithfulness",
    "AuraContract",
    "NodeContract",
    "PropertyVerdict",
    "TaintContract",
    "TaintPath",
    "ContractError",
    # Capability-typed dataflow (injection-proof)
    "analyze_taint",
    "TaintResult",
    "TaintViolation",
    # Risk-controlled abstention
    "RiskController",
    "learn_then_test",
]
