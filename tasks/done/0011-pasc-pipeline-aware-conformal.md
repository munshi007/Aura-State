# 0011: PASC — pipeline-aware conformal

**Status:** backlog
**Type:** feature
**Tags:** `[verification]` `[statistics]` `[frontier]`
**Priority:** later (on-identity polish; follow-on to 0004)
**Depends on:** 0004 (valid conformal foundation)
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** core-hardening research (2026-06-01), frontier-technique survey. Ref: PASC, arXiv:2605.18812.

## Why

Task 0004 makes per-field conformal *valid* but still per-step. An agent is a
*pipeline* of steps; errors compound across nodes, so a 95% guarantee at each
node does not give a 95% guarantee end-to-end. Pipeline-Aware Split Conformal
(PASC) provides coverage over the composed pipeline output — the guarantee
buyers actually care about ("the final answer is calibrated," not "step 3 was").
This is a genuine differentiator: per-pipeline conformal for LLM agents is not in
any mainstream framework.

## What

- Implement PASC over the DAG: calibrate nonconformity at the pipeline-output
  level using a held-out calibration set of full runs, so the interval/abstention
  guarantee applies to the end-to-end result.
- Handle the multiplicity across nodes (compounding) rather than treating each
  node independently.
- Integrate with the abstention policy from task 0012 (pipeline-level risk gate).

## Design

- Calibration set = N complete pipeline executions with ground-truth outputs.
- Nonconformity score defined on the final output (or a per-target composite).
- Threshold via the order-statistic rule from 0004, applied at the pipeline level.
- Expose `PipelineConformal` alongside the per-field `conformal` module; document
  when to use which.

## Test strategy

- empirical end-to-end coverage on a simulated multi-step pipeline >= nominal
- show per-step conformal under-covers end-to-end while PASC meets nominal (the motivating contrast)

`test_pasc_pipeline_coverage_fixes_0011`.

## Acceptance criteria

- [ ] PASC implemented with pipeline-level calibration over full-run held-out data
- [ ] empirical end-to-end coverage >= nominal on the simulated pipeline; per-step-only shown to under-cover
- [ ] integrates with 0012 abstention as a pipeline-level gate
- [ ] docstring cites PASC (arXiv:2605.18812) and the order-statistic basis
- [ ] regression test `test_pasc_*_fixes_0011` passing

## Notes

_record calibration-set construction, nonconformity score, and coverage numbers here_

## Completion (2026-08-24)
`aura_state/verification/pipeline_conformal.py` — `PipelineConformal(confidence).calibrate(predictions, truths)`: split conformal on the composed pipeline-output residual R_i=|pred-truth|; threshold = k-th order statistic, k=ceil((1-alpha)(n+1)), no interpolation; min-n gate (fail-closed, q_hat=inf, calibrated=False) via `min_calibration_samples`. `interval()`, `covers()`, and `should_abstain(action_tolerance)` — the pipeline-level gate composing with 0012. Ref PASC arXiv:2605.18812.
Tests `tests/test_pasc_fixes_0011.py` (3): PASC end-to-end coverage >= nominal; per-step-only demonstrably under-covers (motivating contrast); min-n fails closed; abstention gate. Demo `examples/pasc_demo.py` — **per-step 48.5% vs PASC 90.2% end-to-end coverage** on a 6-step pipeline. Full suite 120 passed.
