# 0021: Capability-containment / least-privilege proof (refinement-type tool envelopes)

**Status:** backlog
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

- [ ] capability manifest (inferred + overridable) per agent/task
- [ ] proves reachable-effect ⊆ manifest; reports each out-of-scope effect with node + reason
- [ ] fail-closed on missing/unknown manifest
- [ ] cites 2605.23951 + names Progent/MiniScope; honest "import" framing
- [ ] tests `test_capability_containment_*_fixes_0021`, passing

## Notes

_record: manifest schema, effect-envelope definition, inference defaults, where we simplify vs the paper._
Relates to [[0020]] (integrity side), [[0023]] (benchmark's least-privilege split), [[0024]] (containment appears in the certificate).
