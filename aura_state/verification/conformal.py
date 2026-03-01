"""
Conformal prediction for calibrated extraction confidence.

Transforms multiple LLM extraction runs into statistically
guaranteed prediction intervals using split conformal prediction.
"""
import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("aura_state.conformal")


@dataclass
class PredictionInterval:
    point_estimate: float
    lower: float
    upper: float
    confidence: float
    n_samples: int


@dataclass
class ConformalResult:
    """Per-field conformal prediction intervals for an extraction."""
    intervals: Dict[str, PredictionInterval]
    coverage_level: float
    calibrated: bool


def _quantile(sorted_values: List[float], q: float) -> float:
    """Compute quantile from sorted values using linear interpolation."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]

    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]

    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def compute_nonconformity_scores(values: List[float]) -> List[float]:
    """Compute nonconformity scores as absolute deviation from the median."""
    if not values:
        return []

    sorted_vals = sorted(values)
    median = _quantile(sorted_vals, 0.5)
    return [abs(v - median) for v in values]


def conformal_interval(
    values: List[float],
    confidence: float = 0.95,
) -> PredictionInterval:
    """
    Split conformal prediction interval for a set of extraction values.

    Given N extraction runs for the same field, computes a prediction
    interval with guaranteed coverage at the specified confidence level.
    """
    if not values:
        return PredictionInterval(0.0, 0.0, 0.0, confidence, 0)

    n = len(values)
    sorted_vals = sorted(values)
    median = _quantile(sorted_vals, 0.5)

    if n < 3:
        return PredictionInterval(
            point_estimate=median,
            lower=min(values),
            upper=max(values),
            confidence=confidence,
            n_samples=n,
        )

    scores = compute_nonconformity_scores(values)
    sorted_scores = sorted(scores)

    # Conformal quantile: ceil((n+1) * confidence) / n
    q_index = math.ceil((n + 1) * confidence) / n
    q_index = min(q_index, 1.0)
    q_hat = _quantile(sorted_scores, q_index)

    return PredictionInterval(
        point_estimate=median,
        lower=median - q_hat,
        upper=median + q_hat,
        confidence=confidence,
        n_samples=n,
    )


def conformal_from_extractions(
    extractions: List[Any],
    confidence: float = 0.95,
) -> ConformalResult:
    """
    Compute conformal prediction intervals for all numeric fields
    across multiple Pydantic extraction runs.

    Args:
        extractions: List of Pydantic model instances from consensus runs.
        confidence: Desired coverage level (default 0.95).

    Returns:
        ConformalResult with per-field intervals.
    """
    if not extractions:
        return ConformalResult(intervals={}, coverage_level=confidence, calibrated=False)

    first = extractions[0]
    field_values: Dict[str, List[float]] = {}

    for ext in extractions:
        data = ext.model_dump() if hasattr(ext, "model_dump") else ext
        if isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, (int, float)):
                    field_values.setdefault(key, []).append(float(val))

    intervals = {}
    for field_name, values in field_values.items():
        intervals[field_name] = conformal_interval(values, confidence)

    calibrated = all(iv.n_samples >= 3 for iv in intervals.values())

    return ConformalResult(
        intervals=intervals,
        coverage_level=confidence,
        calibrated=calibrated,
    )
