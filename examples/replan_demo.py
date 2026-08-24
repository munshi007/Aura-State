#!/usr/bin/env python3
"""
Counterexample-guided replanning: the verifier drives the plan to correct.
==========================================================================

No API key. An unsafe design (untrusted content can reach a send-email sink) is
handed to the repair loop: the taint analysis's counterexample becomes a repair
signal, a sanitizer is inserted, and the plan re-verifies PROVEN. Then a
deliberately unrepairable case shows the loop aborts with the explicit
violation — never a silent pass.

Refs: PAT-Agent (arXiv:2509.23675), VERIMAP (arXiv:2510.17109).

Run:
    python examples/replan_demo.py
"""
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.getLogger("aura_state").setLevel(logging.ERROR)

from aura_state import AuraEngine, Node, CompiledTransition


class Ingest(Node):
    system_prompt = "ingest untrusted content"
    untrusted_source = True

    def handle(self, user_text, extracted_data=None, memory=None):
        return "SendEmail", {}


class SendEmail(Node):
    system_prompt = "send email (irreversible)"
    dangerous_sink = True

    def handle(self, user_text, extracted_data=None, memory=None):
        return "END", {}


def rule(msg):
    print(f"\n{'-' * 68}\n  {msg}\n{'-' * 68}")


def build():
    e = AuraEngine()
    e.register(Ingest, SendEmail)
    e.connect([CompiledTransition(from_node=Ingest, to_node=SendEmail)])
    return e


def main():
    rule("1. Start: untrusted ingest wired straight to SendEmail")
    e = build()
    print(f"  taint verdict: {'PROVEN' if e.analyze_taint().verified else 'VIOLATED'}")

    rule("2. Repair loop (verify → counterexample → repair → re-verify)")
    result = e.repair()
    for it in result.history:
        print(f"    iter {it.iteration}: {it.violation}")
        print(f"             repair: {it.signal.description}  -> applied={it.repaired}")
    print(f"\n  converged: {result.verified}   iterations: {result.iterations}")
    print(f"  final taint verdict: {'PROVEN (safe)' if e.analyze_taint().verified else 'VIOLATED'}")
    print(f"  graph now: " + "  ".join(f"{k}->{v}" for k, v in e._transitions.items() if v))

    rule("3. Unrepairable case aborts explicitly (no silent pass)")
    e2 = build()
    aborted = e2.repair(lambda engine, signal: False, max_iterations=3)
    print(f"  verified: {aborted.verified}")
    print(f"  unresolved: {aborted.unresolved}")

    rule("Result")
    ok = result.verified and not aborted.verified and aborted.unresolved
    print("  The verifier didn't just flag the flaw — it drove the plan to a proven-safe")
    print("  design, and refused to certify the one that couldn't be fixed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
