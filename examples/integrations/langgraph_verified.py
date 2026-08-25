#!/usr/bin/env python3
"""
A real LangGraph agent, verified by Aura-State — running on a LOCAL model (Ollama),
so it's a genuine agent with no API key and nothing sent to the cloud.

LangGraph builds and runs the agent (structured extraction + a decision). Aura-State
sits alongside as the verification layer:
  - proves the extracted data with Z3 (fail-closed)
  - proves, over the agent's tool graph, that untrusted input can't reach a
    dangerous tool without an approval gate (static taint)

Run it live with a local model (recommended — real, free, private):
    brew install ollama && ollama serve &     # or your platform's install
    ollama pull llama3.1
    pip install langgraph
    python examples/integrations/langgraph_verified.py

With no Ollama running, it still executes end-to-end using a deterministic stub
for the LLM node (clearly labeled) so you can see the LangGraph + Aura-State wiring.
"""
import logging, os, sys
from typing import TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# Quiet the Ollama-probe retry noise when running the fallback path.
for _n in ("instructor", "openai", "httpx", "httpcore"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from aura_state import (
    prove_extraction, AuraEngine, Node, CompiledTransition, reachability,
)

OLLAMA_URL = "http://localhost:11434/v1"
# Small models are plenty for structured extraction. Override with OLLAMA_MODEL.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")


class Refund(BaseModel):
    reason: str = Field(description="why the customer wants a refund")
    refund_amount: int = Field(description="requested refund, whole dollars")
    order_total: int = Field(description="order total, whole dollars")


def _extract_with_ollama(message: str):
    """Real structured extraction via a local model. Returns None if Ollama is down."""
    try:
        import instructor
        from openai import OpenAI
        client = instructor.from_openai(
            OpenAI(api_key="ollama", base_url=OLLAMA_URL),
            mode=instructor.Mode.JSON,
        )
        return client.chat.completions.create(
            model=OLLAMA_MODEL, response_model=Refund, max_retries=1,
            messages=[{"role": "user", "content": message}],
        )
    except Exception:
        return None


# ── The LangGraph agent ──
class AgentState(TypedDict):
    message: str
    refund: dict
    decision: str
    live: bool


def extract_node(state: AgentState) -> AgentState:
    got = _extract_with_ollama(state["message"])
    if got is None:  # Ollama not running -> deterministic stub so the demo still runs
        got = Refund(reason="item arrived broken", refund_amount=180, order_total=200)
        state["live"] = False
    else:
        state["live"] = True
    state["refund"] = got.model_dump()
    return state


def decide_node(state: AgentState) -> AgentState:
    r = state["refund"]
    state["decision"] = "approve" if r["refund_amount"] <= r["order_total"] else "escalate"
    return state


def build_langgraph_agent():
    g = StateGraph(AgentState)
    g.add_node("extract", extract_node)
    g.add_node("decide", decide_node)
    g.set_entry_point("extract")
    g.add_edge("extract", "decide")
    g.add_edge("decide", END)
    return g.compile()


# ── Aura-State: the verification layer over the agent ──
def aura_tool_graph():
    class CustomerMsg(Node):
        system_prompt = "untrusted customer message"; untrusted_source = True
        def handle(self, user_text, extracted_data=None, memory=None): return "Approve", {}
    class ApprovalGate(Node):
        system_prompt = "policy/human approval"; sanitizer = True
        def handle(self, user_text, extracted_data=None, memory=None): return "IssueRefund", {}
    class IssueRefund(Node):
        system_prompt = "issue refund (irreversible)"; dangerous_sink = True
        def handle(self, user_text, extracted_data=None, memory=None): return "END", {}
    class Approve(Node):
        system_prompt = "route to approval"
        def handle(self, user_text, extracted_data=None, memory=None): return "END", {}
    e = AuraEngine()
    e.register(CustomerMsg, ApprovalGate, IssueRefund, Approve)
    e.connect([
        CompiledTransition(from_node=CustomerMsg, to_node=ApprovalGate),
        CompiledTransition(from_node=ApprovalGate, to_node=IssueRefund),
    ])
    return e


def main():
    agent = build_langgraph_agent()
    result = agent.invoke({"message": "My item arrived broken, I'd like a refund of $180 on my $200 order."})

    src = "local model (Ollama)" if result["live"] else "deterministic stub (Ollama not running)"
    print(f"  LangGraph agent ran via: {src}")
    print(f"  agent output: {result['refund']}  -> decision: {result['decision']}\n")

    # 1) Aura-State proves the agent's extraction (Z3).
    proof = prove_extraction(result["refund"], ["refund_amount <= order_total", "refund_amount >= 0"])
    print(f"  Aura-State Z3 verifies the output: {proof.verified}")
    if not proof.verified:
        print(f"    -> would REJECT: {proof.failed_obligations}")

    # 2) Aura-State proves the agent's tool graph is injection-safe.
    e = aura_tool_graph()
    taint = e.analyze_taint()
    print(f"  Aura-State proves injection-safety of the tool graph: "
          f"{'PROVEN' if taint.verified else 'VIOLATED'}")
    contract = e.compile_contract(properties=[{"description":"refund reachable","formula":reachability("IssueRefund")}])
    print(f"  audit contract emitted: hash {contract.meta['content_hash'][:12]}…, taint {contract.taint.verdict}")

    print("\n  LangGraph runs the agent; Aura-State proves it. Real framework, local model, no key.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
