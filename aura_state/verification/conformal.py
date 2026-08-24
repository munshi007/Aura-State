"""
Conformal prediction for calibrated extraction confidence.

Transforms multiple LLM extraction runs for the *same field* into a prediction
interval with a distribution-free coverage guarantee.

Estimator: **jackknife+** (Barber, Candès, Ramdas & Tibshirani, 2021,
"Predictive inference with the jackknife+", *Annals of Statistics* 49(1):
486-507). Jackknife+ is chosen over split conformal because N is typically
small here (a handful of consensus runs per field); split conformal would waste
half the samples on a disjoint calibration fold, whereas jackknife+ uses every
point via leave-one-out (LOO) and still yields a finite-sample guarantee.

Coverage identity (jackknife+, their Theorem 1): for exchangeable data the
interval covers a fresh draw with probability >= 1 - 2*alpha in the worst case,
and empirically ~ 1 - alpha for well-behaved (e.g. symmetric) data. The
endpoints are *order statistics* of the LOO-adjusted predictions, NOT
interpolated quantiles:

    k       = ceil((1 - alpha) * (n + 1))          # 1-indexed rank
    upper   = k-th smallest of { mu_{-i} + R_i }
    lower   = (n + 1 - k)-th smallest of { mu_{-i} - R_i }

where mu_{-i} is the LOO point estimate (median of all values except i) and
R_i = |v_i - mu_{-i}| is the LOO nonconformity score.

Minimum sample gate: the k-th order statistic only exists when k <= n, i.e.
    n >= ceil(1 / alpha) - 1   (= 19 at alpha = 0.05, = 9 at alpha = 0.10).
Below this there is NO valid finite-sample threshold: we return the raw
min..max range, set `calibrated = False`, and set `confidence = None`. We never
stamp a nominal coverage label on an uncalibrated interval (fail closed).

SEMANTICS CAVEAT: these values come from re-running extraction on the *same*
input, so the interval characterises the model's **dispersion / self-agreement**
across runs, NOT its error against an external ground truth. A tight interval
means the model is self-consistent, which is necessary but not sufficient for
accuracy.
"""
import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aura_state.conformal")


@dataclass
class PredictionInterval:
    point_estimate: float
    lower: float
    upper: float
    # `confidence` is the nominal coverage level for a calibrated interval, and
    # None when the interval is uncalibrated (too few samples). Kept in the same
    # positional slot for backward compatibility.
    confidence: Optional[float]
    n_samples: int
    calibrated: bool = False
    method: str = "jackknife+"


@dataclass
class ConformalResult:
    """Per-field conformal prediction intervals for an extraction."""
    intervals: Dict[str, PredictionInterval]
    coverage_level: float
    calibrated: bool
    # Field-reporting (defect D): which fields got numeric intervals vs. which
    # were dropped because they were non-numeric.
    covered_fields: List[str] = field(default_factory=list)
    skipped_fields: List[str] = field(default_factory=list)


def min_calibration_samples(alpha: float) -> int:
    """
    Minimum sample count for a valid (1 - alpha) jackknife+ interval.

    Derivation: we need the rank k = ceil((1 - alpha) * (n + 1)) to satisfy
    k <= n so that the k-th order statistic exists. That holds iff
    n >= 1/alpha - 1, i.e. n >= ceil(1/alpha) - 1.
    """
    return math.ceil(1.0 / alpha) - 1


def _median(values: List[float]) -> float:
    """Exact median of a non-empty list (no conformal semantics here)."""
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def compute_nonconformity_scores(values: List[float]) -> List[float]:
    """
    Leave-one-out (LOO) nonconformity scores used by jackknife+.

    For each index i, R_i = |v_i - mu_{-i}| where mu_{-i} is the median of the
    values with index i removed. Using the LOO estimate (rather than the median
    of the full set) is what makes the calibration point disjoint from the point
    being scored — the fix for the "no split" defect (B).
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.0]

    scores: List[float] = []
    for i in range(n):
        rest = values[:i] + values[i + 1:]
        mu_i = _median(rest)
        scores.append(abs(values[i] - mu_i))
    return scores


def conformal_interval(
    values: List[float],
    confidence: float = 0.95,
) -> PredictionInterval:
    """
    Jackknife+ prediction interval for a set of extraction values.

    Given N runs of extraction for the same field, produces an interval that
    covers a fresh draw from the same run-to-run distribution with the nominal
    coverage `confidence` (empirically; worst-case 1 - 2*(1-confidence), per
    Barber et al. 2021). See module docstring for the order-statistic identity
    and the dispersion-vs-error caveat.

    If there are too few samples for a valid threshold
    (n < min_calibration_samples(1 - confidence)), the interval is the raw
    min..max range with `calibrated=False` and `confidence=None`; no nominal
    coverage is claimed.
    """
    alpha = 1.0 - confidence

    if not values:
        return PredictionInterval(
            point_estimate=0.0,
            lower=0.0,
            upper=0.0,
            confidence=None,
            n_samples=0,
            calibrated=False,
        )

    n = len(values)
    point = _median(values)
    min_n = min_calibration_samples(alpha)

    # Defect C: too few samples -> uncalibrated, never a nominal label.
    if n < min_n:
        return PredictionInterval(
            point_estimate=point,
            lower=min(values),
            upper=max(values),
            confidence=None,
            n_samples=n,
            calibrated=False,
        )

    # Jackknife+ (Barber et al. 2021).
    loo_mu: List[float] = []
    resid: List[float] = []
    for i in range(n):
        rest = values[:i] + values[i + 1:]
        mu_i = _median(rest)
        loo_mu.append(mu_i)
        resid.append(abs(values[i] - mu_i))

    upper_pts = sorted(m + r for m, r in zip(loo_mu, resid))
    lower_pts = sorted(m - r for m, r in zip(loo_mu, resid))

    # Defect A: k-th ORDER STATISTIC, no interpolation.
    k = math.ceil((1.0 - alpha) * (n + 1))
    if k > n:
        # Guarded by the min_n gate above, but fail closed if we ever reach here.
        logger.warning("jackknife+ rank k=%d exceeds n=%d; marking uncalibrated", k, n)
        return PredictionInterval(
            point_estimate=point,
            lower=min(values),
            upper=max(values),
            confidence=None,
            n_samples=n,
            calibrated=False,
        )

    upper = upper_pts[k - 1]          # k-th smallest (1-indexed)
    lower = lower_pts[n - k]          # (n + 1 - k)-th smallest (1-indexed)

    return PredictionInterval(
        point_estimate=point,
        lower=lower,
        upper=upper,
        confidence=confidence,
        n_samples=n,
        calibrated=True,
    )


def conformal_from_extractions(
    extractions: List[Any],
    confidence: float = 0.95,
) -> ConformalResult:
    """
    Compute jackknife+ prediction intervals for all numeric fields across
    multiple Pydantic extraction runs.

    Non-numeric fields are recorded in `skipped_fields` rather than silently
    dropped (defect D). A result is `calibrated` only if it has at least one
    field and every field's interval is individually calibrated.

    Args:
        extractions: List of Pydantic model instances from consensus runs.
        confidence: Desired coverage level (default 0.95).

    Returns:
        ConformalResult with per-field intervals plus covered/skipped reporting.
    """
    if not extractions:
        return ConformalResult(
            intervals={},
            coverage_level=confidence,
            calibrated=False,
            covered_fields=[],
            skipped_fields=[],
        )

    field_values: Dict[str, List[float]] = {}
    skipped: List[str] = []

    for ext in extractions:
        data = ext.model_dump() if hasattr(ext, "model_dump") else ext
        if isinstance(data, dict):
            for key, val in data.items():
                # bool is a subclass of int but is categorical, not numeric.
                if isinstance(val, bool):
                    if key not in skipped:
                        skipped.append(key)
                elif isinstance(val, (int, float)):
                    field_values.setdefault(key, []).append(float(val))
                else:
                    if key not in skipped:
                        skipped.append(key)

    intervals: Dict[str, PredictionInterval] = {}
    for field_name, values in field_values.items():
        intervals[field_name] = conformal_interval(values, confidence)

    covered = list(intervals.keys())
    # A field that appeared as numeric in some runs and non-numeric in others is
    # covered (we have numeric samples); drop it from the skipped list.
    skipped = [k for k in skipped if k not in field_values]

    calibrated = bool(intervals) and all(iv.calibrated for iv in intervals.values())

    return ConformalResult(
        intervals=intervals,
        coverage_level=confidence,
        calibrated=calibrated,
        covered_fields=covered,
        skipped_fields=skipped,
    )
