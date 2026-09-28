# 0020: Taint → Information-Flow Control label lattice (non-interference)

**Status:** done (2026-09-28) — integrity half unified; field-level lattice deferred
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

- [x] IFC integrity pass (`verification/ifc.py`) over the graph, labels from the trifecta's `classify_roles` (one role model for both passes)
- [x] integrity non-interference (untrusted must not reach a consequential sink unsanitized), reports the offending flow; confidentiality (private→exfil) remains in `analyze_trifecta` (also role-based) — consistent, not yet merged into one lattice object
- [x] sound / fail-closed: unknown-capability nodes surfaced; a sanitizer is the only declassifier
- [x] parity: benchmark 100%/100%/0 silent misses, full suite 245 green (no regressions). **Closed a real fail-open**: name/data_class/roles-classified untrusted tools are now taint sources (were missed by the capability-only pass)
- [x] `ifc.py` cites AgentFlow (2608.22868) + FIDES (2505.23643); honest static-import framing
- [x] tests `tests/test_ifc_fixes_0020.py` (3) incl. CLI/studio agreement; all passing

## Notes

_record: the lattice definition used, declassification (sanitizer) semantics, field- vs node-level coverage, where we simplify vs the papers._
Relates to [[0014]], [[0023]] (benchmark measures this on imported agents).

## Completion (2026-09-28)
Built `aura_state/verification/ifc.py` — `analyze_ifc(nodes, edges, entry)` runs the **integrity** half of information-flow control (untrusted low-integrity data must not reach a *consequential* sink — external send or local write — without passing a sanitizer/declassifier) on the **same role model** (`classify_roles`) the trifecta uses. Wired both surfaces onto it: `check.py` (CLI) and `server.py /api/verify` (studio) now decide injection-safety identically. This **closed the residual fail-open** the earlier `_cap` patch could not: a tool classified untrusted by name / `data_class` / `roles` (not an explicit `capability`) is now a taint source — verified live (`web_fetch → file_write` now flagged; was "safe"). Parity held (benchmark 100%/100%/0; full suite 245).

**Honest scope / deferred:** delivered the integrity unification + non-interference framing + soundness fix — the core goal. The **confidentiality** half (private→exfil) still lives in `analyze_trifecta` (also role-based, so the two agree) rather than a single merged confidentiality×integrity **label lattice** object, and analysis stays **node-level** (field-level precision deferred). Those are precision/structure refinements, not soundness — good follow-ons. `engine.analyze_field_taint` is now unused by the product paths (left in place; dead-code removal is a separate cleanup).
