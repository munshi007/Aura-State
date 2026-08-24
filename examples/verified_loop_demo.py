#!/usr/bin/env python3
"""
Verified-loop demo: the extraction contract catches a hallucination.
====================================================================

The whole point of Aura-State is that verification runs INSIDE the loop.
This demo proves it end-to-end with no API key: a node declares Z3 proof
obligations, the (mocked) LLM first returns an arithmetically-wrong extraction
(total != area * rate), the loop refuses it with a counterexample and retries,
and the corrected extraction is accepted.

Run:
    python examples/verified_loop_demo.py
"""
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Keep the demo output clean; the loop's retry is shown explicitly below.
logging.getLogger("aura_state").setLevel(logging.ERROR)

from pydantic import BaseModel

from aura_state import AuraEngine, Node
from aura_state.verification.proof_engine import prove_extraction


# ── The data the LLM extracts, and its contract ──

class Quote(BaseModel):
    area: int
    rate: int
    total: int


class PriceQuote(Node):
    system_prompt = "Extract area, rate, and total from the request."
    extracts = Quote
    # These are proven in the loop. An extraction that can't be proven to
    # satisfy them is rejected (fail-closed), not passed downstream.
    obligations = ["total == area * rate", "area > 0", "rate > 0"]

    def handle(self, user_text, extracted_data=None, memory=None):
        return "END", extracted_data.model_dump()


# ── Mock the LLM boundary only: return a WRONG quote first, then a correct one ──

_SCRIPTED = [
    Quote(area=100, rate=3, total=999),   # hallucinated: 100*3 != 999
    Quote(area=100, rate=3, total=300),   # corrected on retry
]


class _Completions:
    def __init__(self):
        self._i = 0

    def create_with_completion(self, **kwargs):
        obj = _SCRIPTED[min(self._i, len(_SCRIPTED) - 1)]
        self._i += 1
        return obj, type("R", (), {"usage": None})()


class _Chat:
    completions = _Completions()


class _MockClient:
    chat = _Chat()


def rule(msg):
    print(f"\n{'-' * 68}\n  {msg}\n{'-' * 68}")


def main():
    rule("1. What Z3 says about the hallucinated extraction, in isolation")
    bad = _SCRIPTED[0].model_dump()
    proof = prove_extraction(bad, PriceQuote.obligations)
    print(f"  extracted : {bad}")
    print(f"  verified  : {proof.verified}")
    print(f"  failed    : {proof.failed_obligations}")
    print(f"  -> 100 * 3 = 300, not 999. The obligation 'total == area * rate' is violated.")

    rule("2. The same node run through the REAL process() loop")
    engine = AuraEngine()
    engine.client = object()                       # mark a live client
    engine.provider.register_client("gpt", _MockClient())
    engine.register(PriceQuote)
    engine._transitions["PriceQuote"] = ["END"]

    next_state, payload = engine.process("PriceQuote", "area 100, rate 3")

    print("  Per-attempt verification (from the loop, not a mock):")
    for m in engine.verification_metrics():
        status = "PASS" if m["passed"] else "REJECT"
        detail = "" if m["passed"] else f"  <- {m['error']}"
        print(f"    attempt {m['attempt']}: {status}{detail}")

    report = engine.verification_reports()[-1]
    print(f"\n  final accepted quote : {payload}")
    print(f"  extraction_verified  : {report['extraction_verified']}")
    print(f"  iterations used      : {report['iterations']}")

    rule("Result")
    ok = report["extraction_verified"] and payload["total"] == payload["area"] * payload["rate"]
    print("  The loop rejected the hallucination, retried, and only accepted an")
    print("  extraction it could PROVE satisfies the contract." if ok else "  UNEXPECTED: contract not enforced.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
