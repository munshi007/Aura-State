#!/usr/bin/env python3
"""
Injection-proof demo: prove untrusted data can't reach a dangerous tool.
========================================================================

Static taint analysis over the typed graph (no API key). An agent ingests
untrusted content and can send email. If untrusted data can reach the
send-email node, the design is VIOLATED and the compiler names the path.
Insert a validation/sanitizer node and the same design becomes PROVEN.

This sells as *impossibility*, not detection: it tracks provenance, not
content, so it can't be fooled by encodings that defeat runtime scanners.

Run:
    python examples/taint_proof_demo.py
"""
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.getLogger("aura_state").setLevel(logging.ERROR)

from aura_state import AuraEngine, Node, CompiledTransition, analyze_taint


class IngestContent(Node):
    system_prompt = "Ingest a document / tool result (untrusted)."
    untrusted_source = True

    def handle(self, user_text, extracted_data=None, memory=None):
        return "SendEmail", {}


class ReviewAndApprove(Node):
    system_prompt = "Human/rule review that sanitizes the content."
    sanitizer = True

    def handle(self, user_text, extracted_data=None, memory=None):
        return "SendEmail", {}


class SendEmail(Node):
    system_prompt = "Send an email (irreversible side effect)."
    dangerous_sink = True

    def handle(self, user_text, extracted_data=None, memory=None):
        return "END", {}


def rule(msg):
    print(f"\n{'-' * 68}\n  {msg}\n{'-' * 68}")


def report(engine):
    r = analyze_taint(engine)
    print(f"  verdict: {'PROVEN (safe)' if r.verified else 'VIOLATED'}")
    for v in r.violations:
        print(f"  untrusted '{v.source}' can reach dangerous '{v.sink}' via: {' -> '.join(v.path)}")
    return r


def main():
    rule("1. Untrusted ingest wired straight to SendEmail")
    unsafe = AuraEngine()
    unsafe.register(IngestContent, SendEmail)
    unsafe.connect([CompiledTransition(from_node=IngestContent, to_node=SendEmail)])
    r1 = report(unsafe)

    rule("2. Same design with a sanitizer (ReviewAndApprove) in the path")
    safe = AuraEngine()
    safe.register(IngestContent, ReviewAndApprove, SendEmail)
    safe.connect([
        CompiledTransition(from_node=IngestContent, to_node=ReviewAndApprove),
        CompiledTransition(from_node=ReviewAndApprove, to_node=SendEmail),
    ])
    r2 = report(safe)

    rule("Result")
    print("  The unsafe design is provably injection-reachable; the sanitized one is")
    print("  provably safe. The verdict compiles into the AuraContract, so a runtime")
    print("  can refuse to deploy a VIOLATED graph.")
    return 0 if (not r1.verified and r2.verified) else 1


if __name__ == "__main__":
    sys.exit(main())
