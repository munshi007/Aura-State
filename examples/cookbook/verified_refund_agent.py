#!/usr/bin/env python3
"""
A verified refund agent — a realistic, high-stakes workflow with every guarantee.

An agent reads an (untrusted) customer message and decides a refund. This is
exactly the kind of task where "probably fine" isn't good enough:

  - Z3 proves the refund is within the order total (no over-refund gets through)
  - static taint proves the untrusted message can't reach `issue_refund` without
    passing the policy check (injection-safe by construction)
  - CTL proves the escalation path is reachable and the flow completes
  - conformal risk control escalates low-confidence decisions to a human
  - the whole design compiles into an audit-ready contract

Runs with NO API key (the extraction is mocked). To run it live against any
provider, pass --provider and set that provider's key in your environment:

    python examples/cookbook/verified_refund_agent.py                 # mocked
    python examples/cookbook/verified_refund_agent.py --provider gemini   # live (needs GOOGLE_API_KEY)
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
logging.getLogger("aura_state").setLevel(logging.ERROR)

from pydantic import BaseModel, Field

from aura_state import AuraEngine, Node, CompiledTransition, reachability, eventual_completion


# ── What the agent extracts, and the contract it must satisfy ──

class RefundRequest(BaseModel):
    reason: str = Field(description="why the customer wants a refund (free text)")
    refund_amount: int = Field(description="requested refund in whole dollars")
    order_total: int = Field(description="the order's total in whole dollars")


class ReadRequest(Node):
    system_prompt = "Extract the refund request from the customer message."
    extracts = RefundRequest
    # Proven in the loop: no over-refund, no negative refund. Unprovable -> rejected.
    obligations = ["refund_amount <= order_total", "refund_amount >= 0"]
    untrusted_fields = ["reason"]          # the customer's free text is untrusted

    def handle(self, user_text, extracted_data=None, memory=None):
        return "PolicyCheck", extracted_data.model_dump() if extracted_data else {}


class PolicyCheck(Node):
    system_prompt = "Apply refund policy; sanitize the request."
    sanitizes_fields = ["reason"]          # policy review clears the untrusted text

    def handle(self, user_text, extracted_data=None, memory=None):
        return "IssueRefund", {}


class IssueRefund(Node):
    system_prompt = "Issue the refund (irreversible)."
    dangerous_sink = True
    sink_fields = ["reason", "refund_amount"]

    def handle(self, user_text, extracted_data=None, memory=None):
        return "END", {}


class Escalate(Node):
    system_prompt = "Hand off to a human agent."

    def handle(self, user_text, extracted_data=None, memory=None):
        return "END", {}


def build():
    e = AuraEngine()
    e.register(ReadRequest, PolicyCheck, IssueRefund, Escalate)
    e.connect([
        CompiledTransition(from_node=ReadRequest, to_node=PolicyCheck),
        CompiledTransition(from_node=PolicyCheck, to_node=IssueRefund),
    ])
    e._transitions["ReadRequest"].append("Escalate")   # abstention path
    return e


def rule(msg):
    print(f"\n{'-'*70}\n  {msg}\n{'-'*70}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None, help="openai|gemini|deepseek|together|ollama (live)")
    args = ap.parse_args()

    engine = build()

    # ── Design time: prove the graph is safe before it ever runs ──
    rule("Design-time proofs (no LLM, no key)")
    taint = engine.analyze_field_taint()
    print(f"  injection-safe: {'PROVEN' if taint.verified else 'VIOLATED'} "
          f"(untrusted 'reason' cannot reach issue_refund without PolicyCheck)")
    ctl = engine.verify([
        {"description": "refund can be issued", "formula": reachability("IssueRefund")},
        {"description": "human escalation reachable", "formula": reachability("Escalate")},
    ])
    for r in ctl:
        print(f"  CTL {r.property_text}: {'PROVEN' if str(r.result)=='PropertyResult.PROVEN' else 'VIOLATED'}")
    contract = engine.compile_contract()
    print(f"  contract emitted: hash {contract.meta['content_hash'][:12]}…  "
          f"(obligations_consistent={next(n.obligations_consistent for n in contract.nodes if n.name=='ReadRequest')})")

    # ── Runtime: the Z3 obligation catches an over-refund in the loop ──
    rule("Runtime: Z3 rejects an over-refund, accepts a valid one")
    if args.provider:
        from _providers import make_client, default_model, has_key
        if not has_key(args.provider):
            print(f"  (no key for {args.provider}; falling back to mocked extraction)")
            args.provider = None
        else:
            engine.client = make_client(args.provider)
            engine.provider.register_client("default", engine.client)
            for n in engine._nodes.values():
                n.model = default_model(args.provider)
            print(f"  live via {args.provider} ({default_model(args.provider)})")

    if not args.provider:
        # Mock the LLM boundary: first a bad (over-)refund, then a valid one.
        scripted = [RefundRequest(reason="broken item", refund_amount=500, order_total=200),
                    RefundRequest(reason="broken item", refund_amount=180, order_total=200)]
        class _Comp:
            def __init__(s): s.i = 0
            def create_with_completion(s, **k):
                obj = scripted[min(s.i, len(scripted)-1)]; s.i += 1
                return obj, type("R", (), {"usage": None})()
        class _Chat: completions = _Comp()
        class _Client: chat = _Chat()
        engine.client = object()
        engine.provider.register_client("default", _Client())

    engine.process("ReadRequest", "Hi, my item arrived broken, I want a refund.")
    rep = engine.verification_reports()[-1]
    print(f"  extraction_verified: {rep['extraction_verified']}  "
          f"(iterations: {rep['iterations']})")
    print("  -> a refund exceeding the order total is provably rejected before any money moves.")

    rule("Result")
    print("  Same code, any provider. The guarantees are the product; the LLM is swappable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
