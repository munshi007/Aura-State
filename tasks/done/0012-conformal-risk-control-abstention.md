# 0012: Conformal Risk Control / Learn-Then-Test abstention

**Status:** done (2026-08-24)
**Type:** feature
**Tags:** `[verification]` `[statistics]` `[frontier]` `[safety]`
**Priority:** now (the procurement story: act only if calibrated risk <= epsilon, else escalate)
**Depends on:** 0004 (valid conformal foundation)
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** core-hardening research (2026-06-01), frontier survey. Refs: Conformal Risk Control arXiv:2208.02814; Learn-Then-Test arXiv:2110.01052.

## Why

Intervals are narrow; the feature high-stakes users want is **"don't act unless
calibrated risk <= epsilon, otherwise escalate to a human."** This reframes
conformal from "here's a range" to a control policy with a guaranteed bound on a
chosen risk (false-action rate, miss rate). Conformal Risk Control (CRC) and
Learn-Then-Test (LTT) give finite-sample guarantees on arbitrary monotone risks
and on tuning multiple thresholds. This is the "knows when it doesn't know"
story made into an actual gate — the single most adoption-relevant feature for
regulated deployments.

## What

- An **abstention policy**: given the conformal output for a step/pipeline, act if
  the calibrated risk is within budget `epsilon`, else **abstain -> escalate**
  (route to a human-in-the-loop node or a safe default action that is explicitly
  configured).
- **CRC** to bound a user-chosen risk function (not just coverage).
- **LTT** to calibrate the threshold(s) with a statistical guarantee when
  multiple knobs are tuned.
- Wire abstention into the engine as a first-class outcome (alongside
  success/failure), surfaced in traces (task 0010) and durable journal (0009).

## Design

- `RiskController` calibrated on a labeled set: picks the largest action region
  whose risk is provably <= epsilon (CRC), with LTT for multi-threshold tuning.
- Engine: after a node's conformal result, consult the controller; on abstain,
  transition to the configured escalation node — never silently guess.
- Compose with PASC (0011) for pipeline-level risk.

## Test strategy

- on a labeled simulation, the realized risk of acted-on decisions <= epsilon within tolerance
- abstention rate moves correctly as epsilon tightens
- an abstain routes to escalation, not a silent default

`test_crc_risk_bound_fixes_0012`, `test_abstention_routes_fixes_0012`.

## Acceptance criteria

- [ ] abstention policy: act iff calibrated risk <= epsilon, else escalate to a configured node (no silent guess)
- [ ] CRC bounds a user-chosen monotone risk; realized risk <= epsilon on the labeled sim
- [ ] LTT used for multi-threshold calibration with its guarantee documented
- [ ] abstention is a first-class engine outcome, surfaced in traces + journal
- [ ] docstrings cite CRC (arXiv:2208.02814) and LTT (arXiv:2110.01052)
- [ ] regression tests `*_fixes_0012` passing

## Notes

_record risk function(s), epsilon defaults, and realized-risk numbers here_

## Completion (2026-08-24)
`aura_state/verification/risk_control.py`:
- `RiskController(epsilon).calibrate(scores, correct)` — Conformal Risk Control (arXiv:2208.02814). Loss L(tau)=1{score>=tau AND wrong}; picks smallest tau with `(n*R_hat+B)/(n+1) <= epsilon`. `can_act=False` (fail-safe abstain-all) when even acting on nobody can't meet epsilon. `should_act`/`should_abstain`.
- `learn_then_test(risk_hats, n, epsilon, delta)` — LTT (arXiv:2110.01052), Bonferroni-corrected Hoeffding p-values; returns family-wise risk-controlled config indices.
Engine: `Node.risk_controller` + `Node.escalation_node` + `Node.risk_score()`; process() STAGE 4b abstains → routes to escalation (first-class outcome, surfaced in `verification_reports()` as `abstained`/`risk_score` and in the trace). Never a silent guess.
Tests `tests/test_risk_control_fixes_0012.py` (4): realized false-action rate 0.026 <= eps=0.05 on held-out sim; abstention rate rises as epsilon tightens; abstain routes to escalation; LTT selects controlled configs. Demo `examples/risk_abstention_demo.py`. Full suite 113 passed.

Open follow-on: compose with PASC (0011) for pipeline-level risk; richer risk functions beyond false-action rate.
