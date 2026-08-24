"""
Pipeline-Aware Split Conformal (PASC): calibrate the END-TO-END output.

Per-field conformal (see `conformal.py`) is valid *per step*, but an agent is a
pipeline and errors compound across nodes: a 95% guarantee at each node does not
give a 95% guarantee on the final answer. PASC calibrates the nonconformity
score on the *composed pipeline output* using a held-out set of complete runs,
so the interval (and the abstention gate built on it) covers the end-to-end
result -- the guarantee people actually care about.

Ref: Pipeline-Aware Split Conformal, arXiv:2605.18812.

Estimator: split conformal on the pipeline-output residual R_i = |pred_i - y_i|
over N complete calibration runs. The threshold is the order statistic

    k     = ceil((1 - alpha) * (n + 1))          # 1-indexed rank, no interpolation
    q_hat = k-th smallest residual

which gives marginal coverage P(|pred - y| <= q_hat) >= 1 - alpha on a fresh
run (Vovk et al. 2005; same order-statistic basis as conformal.py). Split (not
jackknife+) is right here: calibration runs are genuine held-out full executions
with ground truth, and there are usually enough of them. The minimum-n gate
mirrors conformal.py: below n >= ceil(1/alpha) - 1 there is no valid finite-
sample threshold, so we fail closed (calibrated=False, q_hat=inf).
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .conformal import min_calibration_samples


@dataclass
class PipelineConformal:
    """Split conformal calibrated on the composed pipeline output (PASC)."""
    confidence: float = 0.9
    q_hat: Optional[float] = None
    calibrated: bool = False
    n: int = 0

    def calibrate(self, predictions: Sequence[float], truths: Sequence[float]) -> "PipelineConformal":
        """Calibrate on N complete runs: final ``predictions`` vs ground-truth
        ``truths``. Sets the end-to-end threshold ``q_hat``."""
        n = len(predictions)
        if n == 0 or n != len(truths):
            raise ValueError("predictions and truths must be non-empty and equal length")
        alpha = 1.0 - self.confidence
        residuals = sorted(abs(float(p) - float(t)) for p, t in zip(predictions, truths))
        k = math.ceil((1.0 - alpha) * (n + 1))
        self.n = n
        if k > n:
            # Not enough calibration runs for a valid threshold -> fail closed.
            self.calibrated = False
            self.q_hat = math.inf
        else:
            self.calibrated = True
            self.q_hat = residuals[k - 1]
        return self

    def interval(self, prediction: float) -> Tuple[float, float]:
        if self.q_hat is None:
            raise RuntimeError("PipelineConformal used before calibrate()")
        return (prediction - self.q_hat, prediction + self.q_hat)

    def covers(self, prediction: float, truth: float) -> bool:
        if self.q_hat is None:
            raise RuntimeError("PipelineConformal used before calibrate()")
        return abs(prediction - truth) <= self.q_hat

    def min_samples(self) -> int:
        """Calibration runs needed to claim `confidence` end-to-end."""
        return min_calibration_samples(1.0 - self.confidence)

    def should_abstain(self, action_tolerance: float) -> bool:
        """Pipeline-level gate (composes with task 0012 abstention).

        Abstain -> escalate when the calibrated end-to-end uncertainty exceeds
        what the action can tolerate, or when the pipeline is uncalibrated.
        """
        if not self.calibrated:
            return True
        return self.q_hat > action_tolerance
