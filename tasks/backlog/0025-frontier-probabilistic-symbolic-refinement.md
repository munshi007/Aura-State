# 0025: Frontier — probabilistic bounds, symbolic CTL scale, quantified Z3 (umbrella)

**Status:** backlog
**Type:** feature
**Tags:** `[frontier]` `[core]` `[verification]`
**Priority:** later (Phase 2 — optional, after traction)
**Depends on:** 0020, 0021, 0023
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** technique assessment (2026-09-27). These are research-grade upgrades that make the backend genuinely SOTA. Do only if the launch earns it. All framed as **adopting published methods**, not inventing — the field is dense (survey arXiv:2608.14590).

## Why

Three known ceilings in the current stack: CTL doesn't scale (pyModelChecking is explicit-state), deterministic logics ignore that LLMs are stochastic, and Z3 is used for point-checks not universal properties. Each has a published fix.

## What (split into sub-tasks when picked up)

- **Symbolic/scalable model checking**: CTL/LTL via Storm or NuSMV (BDD/IC3-PDR) so verification survives large real agents; add LTL alongside CTL. (Scaling, not novelty.)
- **Probabilistic safety bounds**: model the agent as an MDP from the imported graph, PCTL bound on unsafe-action probability. NB: **ProbGuard already does runtime/learned probabilistic safety** — the open angle is *static-from-structure*; position carefully, no "first" claim.
- **Quantified / refinement-type Z3**: prove obligations over **all** inputs (∀), not one extraction — Dafny/refinement-style. Strengthens the correctness layer.

## Approach

- Each sub-item is its own scoped task when started; prototype on the smallest slice first (repo rule: no backlog→implementation jump).
- Adopt-and-cite: Storm/PRISM, ProbGuard, Dafny — named in code + docs.

## Test strategy

- Per sub-item: adversarial + fail-closed + a scale/coverage assertion; parity with the current engine on existing cases before replacing anything.

## Acceptance criteria

- [ ] (symbolic MC) large-graph CTL/LTL via Storm/NuSMV, parity on current cases, scales past pyModelChecking's ceiling
- [ ] (probabilistic) static MDP→PCTL bound on unsafe-action probability, honestly positioned vs ProbGuard
- [ ] (quantified Z3) ∀-input obligation proofs where decidable, fail-closed otherwise
- [ ] each adopted method cited in code + docs

## Notes

_record: which sub-items were pursued, the tools integrated, honest positioning vs the cited prior work._
Relates to [[0020]], [[0023]]. Split before implementing.
