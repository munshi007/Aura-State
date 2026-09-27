# 0020: Taint → Information-Flow Control label lattice (non-interference)

**Status:** backlog
**Type:** feature
**Tags:** `[core]` `[verification]` `[security]` `[differentiator]`
**Priority:** later (Phase 1 — depth)
**Depends on:** 0014 (capability-typed dataflow), 0019 (honest positioning)
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** competitive verification pass (2026-09-27). Our trifecta/taint is graph **reachability** over node roles. The published SOTA is stronger: AgentFlow (arXiv:2608.22868) and FIDES (arXiv:2505.23643, Microsoft) use a confidentiality×integrity **label lattice** and prove **non-interference**. We are behind on the method — this task adopts it. Our edge stays the *import + product*, not the theory.

## Why

"No path exists" is coarse (false positives, and it isn't a named security property). Upgrading to an IFC label lattice lets us prove **non-interference** (attacker-controlled/untrusted data cannot influence exfil-relevant actions; private data cannot reach untrusted sinks) — a respected guarantee, field-level, with fewer false positives. This hardens the crown-jewel check and lets us honestly say we implement the published IFC method **on the agent you already wrote**, statically, no DSL.

## What

- Replace/augment the reachability taint with an IFC pass over the imported graph:
  - a **confidentiality × integrity lattice** (à la FIDES Fig. 2: (U,H) top … (T,L) bottom), labels on node inputs/outputs derived from capability classes (untrusted/private/exfil/sanitizer) we already infer.
  - propagate labels along edges; a violation = a flow that increases confidentiality-exposure or uses low-integrity data in a consequential (exfil) action without passing a sanitizer (label declassify).
  - report the specific offending flow (source label → sink label · field), not just "a path".
- Keep it **sound / fail-closed**: unknown labels surface as advisories, never silently declassified.

## Approach

- New pass in `aura_state/verification/` (e.g. `ifc.py`) consuming the same node/role model the trifecta uses; keep `trifecta.py` as a compatibility surface or reimplement it on top of IFC.
- Cite AgentFlow + FIDES in the module docstring; document where we match vs simplify (we do static over-approximation on an imported graph; they do dynamic/DSL).
- Field-level where field info exists (extract schemas); node-level otherwise.

## Test strategy

- Adversarial: private→exfil with no sanitizer → non-interference VIOLATED; add a sanitizer (declassify) → PROVEN.
- Integrity: untrusted input feeding a consequential action → VIOLATED; gated → PROVEN.
- Fail-closed: unknown label never declassifies silently.
- Regression: the real-agent corpus verdicts don't get *less* sound than the reachability taint (no new false negatives). `test_ifc_*_fixes_0020`.

## Acceptance criteria

- [ ] IFC lattice pass over imported graphs, labels from existing capability inference
- [ ] proves/refutes **non-interference** (confidentiality + integrity), reports the offending flow
- [ ] sound / fail-closed: unknowns surface, never silent declassify
- [ ] no new false negatives vs the current trifecta on the corpus; fewer false positives where field info exists
- [ ] module cites AgentFlow (2608.22868) + FIDES (2505.23643); honest "static import" framing
- [ ] tests `test_ifc_*_fixes_0020`, real objects, passing

## Notes

_record: the lattice definition used, declassification (sanitizer) semantics, field- vs node-level coverage, where we simplify vs the papers._
Relates to [[0014]], [[0023]] (benchmark measures this on imported agents).
