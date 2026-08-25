#!/usr/bin/env python3
"""
Add Aura-State's proofs to an agent you already built (LangGraph, CrewAI, plain
code — doesn't matter). No rewrite required. No API key.

Positioning: Aura-State is a verification layer, not a replacement orchestrator.
You keep your existing agent; you bolt on the guarantees. Two ways:

  1. Verify the OUTPUT — prove the structured result satisfies your obligations
     (Z3), with a counterexample when it doesn't.
  2. Verify the DESIGN — declare your agent's tool graph and prove properties
     over it (injection-safe dataflow, reachability), and emit a contract.

    python examples/cookbook/verify_existing_agent.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from aura_state import (
    prove_extraction, prove_obligations_satisfiable,
    AuraEngine, Node, CompiledTransition, reachability,
)


# ── Pretend this is YOUR existing agent (LangGraph/CrewAI/whatever). It just
#    returns structured output; Aura-State doesn't need to know how it works. ──
def my_existing_agent(message: str) -> dict:
    # ... your LangGraph graph / CrewAI crew runs here ...
    # For the demo it returns a (deliberately wrong) quote to show the check bite.
    return {"area": 100, "rate": 3, "total": 999}   # 100*3 != 999


def rule(msg):
    print(f"\n{'-'*70}\n  {msg}\n{'-'*70}")


def main():
    # 1) Verify the OUTPUT of the existing agent — no rewrite, just a check.
    rule("1. Verify your agent's output with Z3")
    result = my_existing_agent("quote 100 sqft at $3")
    obligations = ["total == area * rate", "area > 0"]
    proof = prove_extraction(result, obligations)
    print(f"  agent returned: {result}")
    print(f"  verified: {proof.verified}")
    if not proof.verified:
        print(f"  FAILED obligations: {proof.failed_obligations}")
        print("  -> your agent's output is provably wrong; catch it before it ships downstream.")
    # And prove the spec itself is coherent (no self-contradiction):
    print(f"  spec is self-consistent: {prove_obligations_satisfiable(obligations).satisfiable}")

    # 2) Verify the DESIGN — declare your agent's tool topology and prove over it.
    rule("2. Declare your agent's tool graph and prove it's injection-safe")

    class UserInput(Node):
        system_prompt = "untrusted user / retrieved content"
        untrusted_source = True
        def handle(self, user_text, extracted_data=None, memory=None): return "Tool", {}

    class ApprovalGate(Node):
        system_prompt = "human/policy approval"
        sanitizer = True
        def handle(self, user_text, extracted_data=None, memory=None): return "Tool", {}

    class Tool(Node):
        system_prompt = "a tool that spends money / sends data"
        dangerous_sink = True
        def handle(self, user_text, extracted_data=None, memory=None): return "END", {}

    # Model your existing agent's flow: does untrusted input reach the tool
    # WITHOUT going through the approval gate?
    unsafe = AuraEngine()
    unsafe.register(UserInput, Tool)
    unsafe.connect([CompiledTransition(from_node=UserInput, to_node=Tool)])
    print(f"  without an approval gate: {'PROVEN safe' if unsafe.analyze_taint().verified else 'VIOLATED'}")

    safe = AuraEngine()
    safe.register(UserInput, ApprovalGate, Tool)
    safe.connect([
        CompiledTransition(from_node=UserInput, to_node=ApprovalGate),
        CompiledTransition(from_node=ApprovalGate, to_node=Tool),
    ])
    print(f"  with the approval gate:    {'PROVEN safe' if safe.analyze_taint().verified else 'VIOLATED'}")
    contract = safe.compile_contract(properties=[
        {"description": "tool reachable", "formula": reachability("Tool")},
    ])
    print(f"  audit contract for the safe design: hash {contract.meta['content_hash'][:12]}…, "
          f"taint {contract.taint.verdict}")

    rule("Takeaway")
    print("  Keep your agent. Add the proofs. Aura-State verifies output and design")
    print("  without owning your orchestration — see docs/COMPARISON.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
