"""
Regression tests for task 0004: "conformal" must be valid conformal prediction.

CLAUDE.md rule 8/12: the gate is EMPIRICAL COVERAGE against a known
distribution, exercising the real estimator — the quantile is never mocked.

Estimator under test: jackknife+ (Barber et al. 2021). The order-statistic
threshold and min-n gate are validated here, not the interval's shape.
"""
import random

import pytest

from aura_state.verification.conformal import (
    conformal_interval,
    conformal_from_extractions,
    min_calibration_samples,
)
from pydantic import BaseModel


def test_conformal_coverage_fixes_0004():
    """
    Monte-Carlo coverage check.

    Draw calibration points from a known Gaussian, build a jackknife+ interval,
    then draw a fresh point from the SAME distribution and check containment.
    Averaged over many trials the empirical coverage must meet the nominal
    level (within tolerance). This is the real correctness test: a broken
    quantile (interpolation, wrong rank, or same-sample scoring) shows up here
    as under-coverage.
    """
    random.seed(0)

    nominal = 0.90
    tolerance = 0.05
    n = 30              # >= min_calibration_samples(0.10) == 9
    trials = 3000
    mu, sigma = 100.0, 15.0

    assert n >= min_calibration_samples(1.0 - nominal)

    covered = 0
    for _ in range(trials):
        cal = [random.gauss(mu, sigma) for _ in range(n)]
        iv = conformal_interval(cal, confidence=nominal)
        assert iv.calibrated is True
        assert iv.confidence == nominal

        fresh = random.gauss(mu, sigma)
        if iv.lower <= fresh <= iv.upper:
            covered += 1

    empirical = covered / trials
    assert empirical >= nominal - tolerance, (
        f"empirical coverage {empirical:.3f} below nominal {nominal} "
        f"(tolerance {tolerance})"
    )
    # Sanity: a valid interval should not be absurdly conservative either.
    assert empirical <= 0.995


def test_conformal_coverage_at_95_fixes_0004():
    """Same coverage check at the 95% level (min-n gate == 19)."""
    random.seed(1)

    nominal = 0.95
    tolerance = 0.05
    n = 40              # >= min_calibration_samples(0.05) == 19
    trials = 3000
    mu, sigma = 0.0, 1.0

    assert min_calibration_samples(1.0 - nominal) == 19
    assert n >= 19

    covered = 0
    for _ in range(trials):
        cal = [random.gauss(mu, sigma) for _ in range(n)]
        iv = conformal_interval(cal, confidence=nominal)
        assert iv.calibrated is True
        fresh = random.gauss(mu, sigma)
        if iv.lower <= fresh <= iv.upper:
            covered += 1

    empirical = covered / trials
    assert empirical >= nominal - tolerance, (
        f"empirical coverage {empirical:.3f} below nominal {nominal}"
    )


def test_conformal_min_n_fixes_0004():
    """
    Below the min-n gate the interval must declare itself UNCALIBRATED and must
    NOT stamp a nominal confidence label (fail closed, defect C).
    """
    gate_95 = min_calibration_samples(0.05)
    assert gate_95 == 19

    random.seed(2)
    for n in [1, 2, 5, 10, gate_95 - 1]:
        values = [random.gauss(50.0, 5.0) for _ in range(n)]
        iv = conformal_interval(values, confidence=0.95)
        assert iv.calibrated is False, f"n={n} should be uncalibrated"
        assert iv.confidence is None, f"n={n} must not carry a nominal label"
        assert iv.n_samples == n
        # Uncalibrated interval is the raw observed range.
        assert iv.lower == min(values)
        assert iv.upper == max(values)

    # Exactly at the gate it becomes calibrated.
    at_gate = [random.gauss(50.0, 5.0) for _ in range(gate_95)]
    iv = conformal_interval(at_gate, confidence=0.95)
    assert iv.calibrated is True
    assert iv.confidence == 0.95


def test_conformal_order_statistic_no_interpolation_fixes_0004():
    """
    The endpoints must be actual members of the LOO-adjusted point sets (order
    statistics), never interpolated values between them (defect A).
    """
    random.seed(3)
    n = 25
    values = [random.gauss(10.0, 2.0) for _ in range(n)]

    loo_mu, resid = [], []
    for i in range(n):
        rest = values[:i] + values[i + 1:]
        s = sorted(rest)
        m = len(s)
        mid = m // 2
        med = s[mid] if m % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0
        loo_mu.append(med)
        resid.append(abs(values[i] - med))

    upper_pts = {mm + r for mm, r in zip(loo_mu, resid)}
    lower_pts = {mm - r for mm, r in zip(loo_mu, resid)}

    iv = conformal_interval(values, confidence=0.9)
    assert iv.upper in upper_pts
    assert iv.lower in lower_pts


def test_conformal_reports_skipped_fields_fixes_0004():
    """
    Non-numeric fields must be reported as skipped, not silently dropped
    (defect D), and numeric fields recorded as covered.
    """
    class Extraction(BaseModel):
        budget: float
        rooms: int
        city: str          # non-numeric -> skipped
        verified: bool     # categorical -> skipped

    extractions = [
        Extraction(budget=1000.0 + i, rooms=3, city="NYC", verified=True)
        for i in range(20)
    ]
    result = conformal_from_extractions(extractions, confidence=0.95)

    assert "budget" in result.covered_fields
    assert "rooms" in result.covered_fields
    assert "city" in result.skipped_fields
    assert "verified" in result.skipped_fields
    # Skipped fields never sneak into intervals.
    assert "city" not in result.intervals
    assert "verified" not in result.intervals
