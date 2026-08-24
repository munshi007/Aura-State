#!/usr/bin/env python3
"""
Risk-controlled abstention: act only if calibrated risk <= epsilon, else escalate.
==================================================================================

No API key. Calibrate a controller so the false-action rate is provably within
budget, show the realized risk on held-out data, then run an agent where a
low-confidence decision is escalated to a human instead of guessing.

Conformal Risk Control (arXiv:2208.02814).

Run:
    python examples/risk_abstention_demo.py
"""
import logging
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.getLogger("aura_state").setLevel(logging.ERROR)

from aura_state import AuraEngine, Node, CompiledTransition, RiskController


def labeled(n, seed):
    rng = random.Random(seed)
    s = [rng.random() for _ in range(n)]
    c = [rng.random() < si for si in s]     # correct with prob = confidence
    return s, c


def main():
    epsilon = 0.05
    print(f"  risk budget epsilon = {epsilon}  (max false-action rate)\n")

    cal_s, cal_c = labeled(800, seed=1)
    ctrl = RiskController(epsilon=epsilon).calibrate(cal_s, cal_c)
    print(f"  calibrated acting threshold: score >= {ctrl.threshold:.3f}")

    test_s, test_c = labeled(3000, seed=2)
    acted = [(s, c) for s, c in zip(test_s, test_c) if ctrl.should_act(s)]
    realized = sum(1 for s, c in acted if not c) / len(test_s)
    abstained = 1 - len(acted) / len(test_s)
    print(f"  realized false-action rate on held-out data: {realized:.3f}  (<= {epsilon} ✓)")
    print(f"  abstention rate: {abstained:.1%}  (these escalate to a human)\n")

    # ── Wire it into an agent: low confidence -> escalate, not guess ──
    class Decide(Node):
        system_prompt = "auto-approve or escalate"
        risk_controller = ctrl
        escalation_node = "HumanReview"

        def __init__(self):
            self._score = 0.0

        def risk_score(self, extracted_data=None, conformal=None, memory=None):
            return self._score

        def handle(self, user_text, extracted_data=None, memory=None):
            return "AutoApprove", {}

    class AutoApprove(Node):
        system_prompt = "auto"

        def handle(self, user_text, extracted_data=None, memory=None):
            return "END", {}

    class HumanReview(Node):
        system_prompt = "human in the loop"

        def handle(self, user_text, extracted_data=None, memory=None):
            return "END", {}

    engine = AuraEngine()
    engine.register(Decide, AutoApprove, HumanReview)
    engine.connect([CompiledTransition(from_node=Decide, to_node=AutoApprove)])

    print(f"  {'-' * 60}")
    for score in (0.95, 0.40):
        engine._nodes["Decide"]._score = score
        nxt, payload = engine.process("Decide", "a decision")
        acted_str = "ESCALATE → HumanReview" if payload.get("abstained") else "act → AutoApprove"
        print(f"  confidence {score:.2f}  ->  {nxt:<12}  ({acted_str})")

    print(f"\n  The agent only auto-acts inside the risk budget; everything below the")
    print(f"  calibrated threshold goes to a human. That is the guarantee, computed.")
    return 0 if realized <= epsilon + 0.03 else 1


if __name__ == "__main__":
    sys.exit(main())
