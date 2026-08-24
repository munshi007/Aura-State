#!/usr/bin/env python3
"""
Pipeline-Aware Split Conformal: calibrate the final answer, not just step 3.
============================================================================

No API key. A 6-step pipeline compounds a little error at each node. Per-step
conformal is valid at each step yet UNDER-COVERS the end-to-end output; PASC
calibrates on the composed output and meets the nominal guarantee.

Ref: PASC, arXiv:2605.18812.

Run:
    python examples/pasc_demo.py
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aura_state import PipelineConformal

K, SIGMA, CONF = 6, 1.0, 0.90


def runs(n, seed):
    rng = random.Random(seed)
    T, P, S = [], [], []
    for _ in range(n):
        truth = rng.uniform(0, 100)
        errs = [rng.gauss(0, SIGMA) for _ in range(K)]
        T.append(truth); P.append(truth + sum(errs)); S.append(abs(errs[0]))
    return T, P, S


def main():
    print(f"  {K}-step pipeline, per-step noise sigma={SIGMA}, target coverage {CONF:.0%}\n")
    cal_t, cal_p, cal_s = runs(400, 1)
    test_t, test_p, _ = runs(4000, 2)

    pasc = PipelineConformal(confidence=CONF).calibrate(cal_p, cal_t)
    pasc_cov = sum(pasc.covers(p, t) for p, t in zip(test_p, test_t)) / len(test_t)

    s = sorted(cal_s); k = math.ceil(CONF * (len(s) + 1)); q_step = s[min(k, len(s)) - 1]
    step_cov = sum(1 for p, t in zip(test_p, test_t) if abs(p - t) <= q_step) / len(test_t)

    print(f"  per-step conformal  interval ±{q_step:5.2f}   end-to-end coverage {step_cov:5.1%}   ← under-covers")
    print(f"  PASC (end-to-end)   interval ±{pasc.q_hat:5.2f}   end-to-end coverage {pasc_cov:5.1%}   ← meets {CONF:.0%}")
    print(f"\n  Each step was calibrated. The final answer wasn't — until PASC.")
    print(f"  It composes with abstention: escalate when the end-to-end interval is")
    print(f"  wider than the action tolerates.  (RiskController / should_abstain)")
    return 0 if (pasc_cov >= CONF - 0.03 and step_cov < CONF - 0.05) else 1


if __name__ == "__main__":
    sys.exit(main())
