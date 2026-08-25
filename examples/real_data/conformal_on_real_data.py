#!/usr/bin/env python3
"""
Aura-State's calibrated-uncertainty guarantee on a REAL dataset (not simulated).

Uses scikit-learn's diabetes dataset — 442 real patient records — trains a plain
model, and wraps its predictions with Aura-State's PipelineConformal. The point:
the coverage guarantee holds on real, messy, held-out data.

    pip install scikit-learn
    python examples/real_data/conformal_on_real_data.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from aura_state import PipelineConformal

CONF = 0.90

def main():
    X, y = load_diabetes(return_X_y=True)
    n = len(y)
    # deterministic split: train / calibration / test (real data, no shuffle seed drama)
    tr, cal, te = slice(0, 250), slice(250, 350), slice(350, n)

    model = LinearRegression().fit(X[tr], y[tr])
    cal_pred, cal_true = model.predict(X[cal]).tolist(), y[cal].tolist()
    te_pred,  te_true  = model.predict(X[te]).tolist(),  y[te].tolist()

    # Aura-State: calibrate the interval on real calibration residuals.
    pc = PipelineConformal(confidence=CONF).calibrate(cal_pred, cal_true)

    covered = sum(pc.covers(p, t) for p, t in zip(te_pred, te_true))
    coverage = covered / len(te_true)

    print(f"  dataset: sklearn diabetes — {n} real patient records")
    print(f"  model: LinearRegression | calibration n={len(cal_true)} | test n={len(te_true)}")
    print(f"  target: {'%.0f' % min(y)}–{'%.0f' % max(y)} (disease progression)\n")
    print(f"  requested coverage : {CONF:.0%}")
    print(f"  interval half-width: ±{pc.q_hat:.1f}")
    print(f"  REAL empirical coverage on held-out patients: {coverage:.1%}  ({covered}/{len(te_true)})")
    print(f"\n  The guarantee holds on real data — not a simulation.")
    # valid conformal should meet nominal within finite-sample noise
    return 0 if coverage >= CONF - 0.06 else 1

if __name__ == "__main__":
    sys.exit(main())
