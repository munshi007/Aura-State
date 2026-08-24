"""Task 0011: Pipeline-Aware Split Conformal covers the END-TO-END output,
where per-step-only conformal under-covers because errors compound."""
import math
import random

from aura_state import PipelineConformal


K_STEPS = 6          # a 6-node pipeline
SIGMA = 1.0          # per-step noise scale
CONF = 0.9


def _runs(n, seed):
    """Each run: known truth; the final prediction accumulates K per-step
    errors, so the end-to-end error is much larger than any single step's."""
    rng = random.Random(seed)
    truths, preds, single_step_resid = [], [], []
    for _ in range(n):
        truth = rng.uniform(0, 100)
        step_errs = [rng.gauss(0, SIGMA) for _ in range(K_STEPS)]
        pred = truth + sum(step_errs)           # compounded end-to-end error
        truths.append(truth)
        preds.append(pred)
        single_step_resid.append(abs(step_errs[0]))   # what a per-step view sees
    return truths, preds, single_step_resid


def test_pasc_pipeline_coverage_fixes_0011():
    # Calibrate PASC on full runs; measure end-to-end coverage on held-out runs.
    cal_t, cal_p, cal_step = _runs(400, seed=1)
    pasc = PipelineConformal(confidence=CONF).calibrate(cal_p, cal_t)
    assert pasc.calibrated

    test_t, test_p, _ = _runs(4000, seed=2)
    pasc_cov = sum(1 for p, t in zip(test_p, test_t) if pasc.covers(p, t)) / len(test_t)

    # PASC meets nominal end-to-end (small finite-sample tolerance).
    assert pasc_cov >= CONF - 0.03

    # Per-step-only threshold: calibrate q on single-step residuals, apply to the
    # composed output. It under-covers, because the end-to-end error is ~sqrt(K)
    # larger than a single step.
    step_sorted = sorted(cal_step)
    n = len(step_sorted)
    k = math.ceil(CONF * (n + 1))
    q_step = step_sorted[min(k, n) - 1]
    per_step_cov = sum(1 for p, t in zip(test_p, test_t) if abs(p - t) <= q_step) / len(test_t)

    assert per_step_cov < CONF - 0.05          # demonstrably under-covers
    assert pasc_cov - per_step_cov > 0.1        # PASC is materially better
    assert pasc.q_hat > q_step                  # PASC's interval is appropriately wider


def test_pasc_min_n_fails_closed_fixes_0011():
    t, p, _ = _runs(5, seed=3)                   # far below the gate
    pasc = PipelineConformal(confidence=0.95).calibrate(p, t)
    assert pasc.calibrated is False
    assert pasc.q_hat == math.inf
    assert pasc.should_abstain(action_tolerance=10.0) is True   # uncalibrated -> abstain


def test_pasc_abstention_gate_fixes_0011():
    # Composes with 0012: abstain when end-to-end uncertainty exceeds tolerance.
    t, p, _ = _runs(400, seed=4)
    pasc = PipelineConformal(confidence=CONF).calibrate(p, t)
    assert pasc.should_abstain(action_tolerance=pasc.q_hat / 2) is True    # too uncertain
    assert pasc.should_abstain(action_tolerance=pasc.q_hat * 2) is False   # within budget
