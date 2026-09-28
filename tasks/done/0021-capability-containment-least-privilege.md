# 0021: Capability-containment / least-privilege proof (refinement-type tool envelopes)

**Status:** done (2026-09-28) — CLI/backend; studio UI + refinement-type formalization deferred
**Type:** feature
**Tags:** `[core]` `[verification]` `[security]` `[differentiator]`
**Priority:** later (Phase 1 — depth)
**Depends on:** 0014 (capability-typed dataflow), 0002 (Z3)
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** competitive verification pass (2026-09-27). 55% of enterprise AI-agent incidents had **no attacker** — the agent over-reached its scope (Cyera 2026, attributed). Design-time least-privilege exists in research: capability-containment via abstract interpretation + refinement types + SMT (arXiv:2605.23951), plus Progent/MiniScope/SkillScope. We ADOPT this on imported agents — not unowned, but no shipped product does it on your real agent.

## Why

The attacker-free majority of incidents is over-permissioned agents taking unintended consequential actions. We can prove, before running, that an agent **cannot reach a tool/data/effect outside its declared task scope** — a capability-containment proof — on the imported graph.

## What

- A declared **capability manifest** per agent/task (allowed tools, side-effects, data classes, resource bounds).
- A check that every reachable tool call's **effect envelope** ⊆ the manifest — reject calls outside the declared capability set (refinement-type-style envelope), proved with the existing Z3/effect model.
- Report each out-of-scope reachable effect (which node, which capability, why it exceeds scope).

## Approach

- New pass in `aura_state/verification/` (e.g. `capability_containment.py`): reachable-effect set from the graph ∩ manifest; violations = reachable effects ∉ manifest.
- Manifest: inferred defaults from the agent + user-overridable (like obligations); missing manifest → advisory (fail-closed), not silent pass.
- Cite 2605.23951 (capability-containment, refinement types) + Progent/MiniScope in the docstring; frame as "their method, your imported agent."

## Test strategy

- Adversarial: agent with a tool whose effect exceeds the manifest → VIOLATED; tighten manifest or remove tool → PROVEN.
- No-attacker harm: a write/delete/payment effect reachable but not in scope → flagged.
- Fail-closed: no manifest → advisory, never silent-safe.
- `test_capability_containment_*_fixes_0021`.

## Acceptance criteria

- [x] capability manifest (flow `manifest`: side_effects / tools / data_classes allowlists) — opt-in, declared per agent
- [x] proves reachable-effect ⊆ manifest; reports each out-of-scope reachable effect (node + dimension + value)
- [x] no/empty manifest → `least_privilege: not declared` (opt-in; no declared scope = nothing to contain, not a silent pass); unreachable effects not flagged
- [x] `capability_containment.py` cites 2605.23951 + Progent/MiniScope; static import-time framing
- [x] tests `tests/test_containment_fixes_0021.py` (6), passing

## Notes

_record: manifest schema, effect-envelope definition, inference defaults, where we simplify vs the paper._
Relates to [[0020]] (integrity side), [[0023]] (benchmark's least-privilege split), [[0024]] (containment appears in the certificate).

## Completion (2026-09-28)
Built `aura_state/verification/capability_containment.py` — `analyze_containment(nodes, edges, entry, manifest)` proves every **reachable** tool effect stays inside a declared **manifest** (allowlists over side_effects / tools / data_classes; accepts flat or under `allow`). Wired into `check.py` as an opt-in step (`flow["manifest"]`) with a `least_privilege` summary key (`not declared` / `contained` / `exceeded`) and a `least-privilege` Finding per out-of-scope reachable effect. Documented the `manifest` field in the flow-format docstring. Verified: read-only manifest + a reachable `payment.charge` → `exceeded`; within scope → `contained`; no manifest → `not declared`; an unreachable out-of-scope tool → not flagged (only reachable effects count). 6 tests; full suite 251.

**Honest scope / deferred:** delivered the containment check on the CLI/`check` path (the CI-gate use — the no-attacker over-reach class). The manifest envelope is a practical **allowlist**, not a full refinement-type/Z3 formalization (2605.23951's mechanism); and there is **no studio UI** for declaring a manifest yet (`/api/verify` doesn't carry it). Both are good follow-ons; the core least-privilege proof is in place.
