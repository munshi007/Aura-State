"""
Risk-controlled abstention: act only if calibrated risk <= epsilon, else escalate.

Reframes calibrated uncertainty from "here is a range" into a control policy
with a finite-sample guarantee on a chosen risk. A node emits a confidence
score per decision; the controller, calibrated on a labeled set, acts only when
the score clears a threshold chosen so the *false-action rate* is provably
<= epsilon, and otherwise abstains -> escalate to a human. Never a silent guess.

Method: Conformal Risk Control (Angelopoulos et al., "Conformal Risk Control",
arXiv:2208.02814). For a loss that is monotone non-increasing in the acting
threshold tau -- here L_i(tau) = 1{ score_i >= tau AND the action is wrong } --
the empirical risk R_hat(tau) is non-increasing, and

    tau_hat = inf { tau : (n * R_hat(tau) + B) / (n + 1) <= epsilon }

(with loss bound B) guarantees E[ L_test(tau_hat) ] <= epsilon. We pick the
*smallest* such tau: the most permissive threshold whose false-action rate is
still controlled, so the agent acts as often as the risk budget allows.

`learn_then_test` (Angelopoulos et al., "Learn Then Test", arXiv:2110.01052)
is provided for calibrating *several* thresholds/configs at once with a
family-wise guarantee via Bonferroni-corrected Hoeffding p-values.
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass
class RiskController:
    """Calibrated act/abstain gate with a bound on the false-action rate."""
    epsilon: float                       # risk budget (max false-action rate)
    loss_bound: float = 1.0              # B: sup of the per-point loss (1 for 0/1)
    threshold: Optional[float] = None    # calibrated acting threshold (act iff score >= threshold)
    calibrated: bool = False
    can_act: bool = True                 # False if even abstaining-all can't meet epsilon

    def calibrate(self, scores: Sequence[float], correct: Sequence[bool]) -> "RiskController":
        """Calibrate on a labeled set.

        ``scores[i]`` in [0,1] is the model's confidence for decision i (higher
        = more confident); ``correct[i]`` is whether acting on i would be right.
        Chooses the smallest threshold whose CRC-corrected false-action rate is
        <= epsilon.
        """
        n = len(scores)
        if n == 0 or n != len(correct):
            raise ValueError("scores and correct must be non-empty and equal length")

        # Candidate thresholds: each observed score (act iff score >= tau), plus
        # one above the max (never act). Ascending, so the first that meets the
        # bound is the smallest / most permissive.
        candidates = sorted(set(scores)) + [max(scores) + 1.0]
        chosen: Optional[float] = None
        for tau in candidates:
            wrong_actions = sum(1 for s, c in zip(scores, correct) if s >= tau and not c)
            r_hat = wrong_actions / n
            r_plus = (n * r_hat + self.loss_bound) / (n + 1)
            if r_plus <= self.epsilon:
                chosen = tau
                break

        if chosen is None:
            # Even acting on nobody leaves R_plus = B/(n+1) > epsilon: cannot
            # meet the budget with this much data. Fail safe: always abstain.
            self.can_act = False
            self.threshold = math.inf
        else:
            self.can_act = True
            self.threshold = chosen
        self.calibrated = True
        return self

    def should_act(self, score: float) -> bool:
        if not self.calibrated:
            raise RuntimeError("RiskController used before calibrate()")
        if not self.can_act:
            return False
        return score >= self.threshold

    def should_abstain(self, score: float) -> bool:
        return not self.should_act(score)


def learn_then_test(
    risk_hats: Sequence[float],
    n: int,
    epsilon: float,
    delta: float = 0.05,
) -> List[int]:
    """Learn-Then-Test: which configs are risk-controlled at level epsilon?

    Given the empirical risk of each of ``m`` candidate configs on a shared
    calibration set of size ``n`` (losses in [0,1]), test each null
    "H_j: risk_j > epsilon" with a Hoeffding p-value and reject (declare config
    j risk-controlled) at Bonferroni level ``delta/m``. Returns the indices of
    configs with a family-wise guarantee that their true risk <= epsilon.
    """
    m = len(risk_hats)
    if m == 0:
        return []
    controlled: List[int] = []
    for j, r_hat in enumerate(risk_hats):
        # One-sided Hoeffding p-value for H0: risk > epsilon.
        if r_hat >= epsilon:
            p = 1.0
        else:
            p = math.exp(-2 * n * (epsilon - r_hat) ** 2)
        if p <= delta / m:
            controlled.append(j)
    return controlled
