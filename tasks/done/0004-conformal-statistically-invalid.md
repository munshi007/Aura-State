# 0004: Conformal statistically invalid -> split + order-statistic

**Status:** done (2026-08-24)
**Type:** correctness
**Tags:** `[verification]` `[statistics]` `[soundness]`
**Priority:** now
**Depends on:** none
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** core-hardening research (2026-06-01), conformal audit. The "statistically guaranteed prediction interval" claim does not hold as implemented.

## Why

Conformal prediction is the calibrated-uncertainty pillar — the "knows when it
doesn't know" guarantee. The module promises "statistically guaranteed coverage."
As written it is not split conformal, uses the wrong quantile, and silently
drops data, so the 95% number is marketing, not a guarantee. Shipping a false
coverage claim is a credibility (and, in regulated use, liability) risk.

## What's broken

`aura_state/verification/conformal.py`.

**A — wrong conformal quantile (conformal.py:90-92).**

```python
q_index = math.ceil((n + 1) * confidence) / n      # :90  -- index used as a fraction
q_index = min(q_index, 1.0)                          # :91
q_hat = _quantile(sorted_scores, q_index)            # :92  -- linear interpolation
```

The valid split-conformal threshold is the **k-th order statistic** of the
calibration scores with `k = ceil((n+1)*(1-alpha))` (here `alpha = 1-confidence`),
taken **directly** (no interpolation). This code computes a *fraction*
`ceil((n+1)*conf)/n`, feeds it to an interpolating `_quantile` (conformal.py:32-47),
and interpolates between order statistics — which breaks the finite-sample
coverage guarantee.

**B — no split (conformal.py:50-57, :75, :86).** The median used as the point
estimate (conformal.py:75) and the nonconformity scores (conformal.py:50-57,
`abs(v - median)`) are computed from the **same** values. Split conformal requires
the calibration scores to be computed against a point estimate from a **disjoint**
fold (or via a jackknife+ / full-conformal construction). Using one set for both
makes the residuals optimistic and voids the guarantee.

**C — too-few-samples silently "works" (conformal.py:77-84).** For `n < 3` it
returns `min..max` and still stamps `confidence`. Valid 95% coverage needs
`n >= ceil(1/alpha) - 1 = 19`. Below that the result must be flagged
**uncalibrated**, not returned with a confidence label.

**D — silent field drop (conformal.py:127-129).** Only `int/float` fields are
kept; every other field vanishes with no flag. Callers cannot tell a field was
dropped vs covered.

**E — wrong semantics of "coverage."** Re-running the same input N times and
taking spread measures *model dispersion / self-consistency*, not error vs
ground truth. Exchangeability holds only across same-input re-runs; this cannot
be sold as accuracy on new inputs. Must be documented; ideally calibrated against
a labeled set.

## Repro

```python
from aura_state.verification.conformal import conformal_interval
import random
# Empirical coverage falls short of nominal for moderate n.
# Build many calibration/test draws, compute the fraction of test points
# inside [lower, upper]; with the interpolated wrong-index quantile it
# undershoots the claimed 0.95.
```

(Quantify in scope: simulate a known distribution, compare empirical coverage to
nominal across n = 5, 20, 50, 200.)

## Root cause

The implementation reaches for a percentile helper (`_quantile`, interpolation)
where conformal theory requires a specific *order statistic*, and it skips the
train/calibration split that makes the scores valid. Classic "looks like
conformal, isn't."

## Fix

- **Order-statistic threshold.** `k = ceil((n+1)*(1-alpha))`; if `k > n`, the
  interval is the full range and the result is flagged uncalibrated.
  `q_hat = sorted_scores[k-1]` (1-indexed k -> 0-indexed `k-1`). No interpolation.
- **Split (or jackknife+).** Hold out a calibration fold disjoint from the point
  being covered. For small N where a split wastes data, implement **jackknife+**
  (leave-one-out residuals) and document which estimator is active.
- **Minimum-n gate.** `n >= ceil(1/alpha) - 1` to claim `1-alpha` coverage; else
  return `calibrated=False` with `confidence=None`/explicit "uncalibrated".
- **No silent drops.** Record which fields were covered, which were skipped
  (non-numeric), and why, on `ConformalResult`.
- **Honest semantics.** Docstring + result clearly state this is same-input
  dispersion unless calibrated against labeled ground truth.

## Test strategy

Empirical coverage is the gate (CLAUDE.md rule 8): simulate a known distribution,
generate many calibration/test splits, assert mean empirical coverage of the
produced intervals `>= nominal - tolerance` for `n >= 19`, and assert
`calibrated=False` for `n < 19`. No mock of the quantile. Add
`test_conformal_coverage_fixes_0004`, `test_conformal_min_n_fixes_0004`.

## Acceptance criteria

- [ ] threshold is the k-th order statistic, `k = ceil((n+1)*(1-alpha))`, no interpolation
- [ ] calibration scores use a disjoint split or documented jackknife+; point estimate not drawn from the same fold as its own residuals
- [ ] `n < ceil(1/alpha)-1` returns `calibrated=False` (no confidence label), never `min..max` stamped at nominal
- [ ] non-numeric / skipped fields are reported on the result, not dropped silently
- [ ] empirical-coverage test passes (`>= nominal - tol`) for n >= 19; uncalibrated flagged below
- [ ] docstring states the same-input-dispersion semantics and cites the order-statistic identity
- [ ] regression tests `test_conformal_*_fixes_0004` added, executing the real estimator

## Notes

_record chosen estimator (split vs jackknife+), tolerance, and the coverage-simulation numbers here_

## Completion (2026-08-24)
Rewritten as valid **jackknife+** (Barber et al. 2021). Order-statistic threshold `k = ceil((1-alpha)(n+1))`, no interpolation. Min-n gate `ceil(1/alpha)-1` (=19@95%) → below it returns `calibrated=False`, `confidence=None`. New fields: `calibrated`, `method`, `covered_fields`, `skipped_fields` (no silent drops). Docstring states same-input-dispersion semantics. Empirical coverage sim: nominal 0.90→0.906 (n=30), 0.95→0.949 (n=40). Tests: `tests/test_conformal_fixes_0004.py` (5). Full suite 105 passed.
