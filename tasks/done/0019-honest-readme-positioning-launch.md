# 0019: Honest README rewrite + positioning + launch assets

**Status:** done (2026-09-28)
**Type:** docs
**Tags:** `[launch]` `[positioning]` `[research]`
**Priority:** now (Phase 0 — launch)
**Depends on:** 0016 (demo output to show)
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** competitive verification pass (2026-09-27). Reading the primaries (AgentProof, AgentFlow, FIDES arXiv:2505.23643, capability-containment arXiv:2605.23951, survey arXiv:2608.14590) demolished the "unowned technique" thesis: the methods all exist in research prototypes. The honest, defensible moat is **productization + import + breadth**, not novelty. The current README both undersells the 5 checks (reads as a prompt-injection box) and would overclaim novelty on HN.

## Why

The README is the #1 conversion surface and it must be **HN-proof**: no "first/only/no-one-does-X" claims that a cited paper refutes. Position on what's true — aura-state is the *shipped tool that ingests the agent you already wrote* and runs multiple design-time checks, while the rigorous methods live in prototypes that make you rewrite into a DSL/planner.

## What

- Rewrite README to lead with: the 5 checks (safe/correct/live/calibrated/governed), the one-move story (import your real agent → verdict + fix + certificate), and **honest citations** — name AgentProof (topology only), AgentFlow (dataflow but its own DSL), FIDES (runtime), capability-containment (design-time but a prototype), and state plainly what aura-state adds: **import of real LangGraph/CrewAI/AutoGen/MCP agents + breadth + shipped product**.
- Real receipts: the CrewAI/LangGraph/AutoGen scans we already ran (0.11.2 work), with the actual findings.
- Launch assets: 30-sec demo GIF (import→finding→fix→proof), and HN/PH/LinkedIn copy built on a real found-vulnerability story — all attribution-honest (e.g. cite Cyera stats as "per Cyera 2026", not ground truth).

## Approach

- Draft README + assets; every comparative claim traces to a parseable primary source (hard rule after the FIDES fetch fabrication — see Notes).
- Keep the "no fabricated numbers" gate: vendor stats attributed, technique claims cited.

## Test strategy

- N/A (docs) — review gate: every comparative/novelty claim has a source link; a skeptical reader with the cited papers can't falsify a sentence.

## Acceptance criteria

- [x] README leads with the 5 checks + the import/verdict/fix/certificate story (`aura-state demo` output → 5-proofs table → "point at your own agent")
- [x] honest competitive framing with citations — comparison table linking AgentProof / AgentFlow / FIDES + the survey; states our edge as import + breadth
- [x] no "first/only/unowned-technique" claims; moat stated as product + import + breadth + sound/fail-closed
- [x] real scan receipts included (the existing 5/6 lethal-trifecta audit); stats attributed (Cyera 2026 in launch copy)
- [x] HN/PH/LinkedIn copy drafted (`docs/LAUNCH.md`); **demo GIF** recorded (`assets/demo.gif`) — Playwright walkthrough of the front door → import → finding → shareable proof, assembled with PIL (5-frame slideshow; no ffmpeg for smooth video — re-record with screen capture later if wanted)

## Notes

_record: final positioning line; the citation list; which stats are attributed vs dropped._
**Hard rule (why):** a WebFetch on a binary PDF fabricated a "FIDES is static + imports LangChain" answer that the real PDF refuted. No claim ships without a parseable primary read.
Relates to [[0016]] (demo), [[0023]] (benchmark backs the story later).

## Completion (2026-09-27)
Rewrote the README hero: tagline broadened from "can't be prompt-injected" to "injection-safe, in-spec, and won't act out of bounds — a design-time proof over the agent you already wrote, not a runtime guardrail." New sections: **One command, zero setup** (`aura-state demo` output), **Five things it proves** (Safe/Correct/Live/Calibrated/Governed table — "prompt injection is only one of them"), **Point it at your own agent**, and **How this is different (and honest)** — a comparison table crediting AgentProof (topology only) / AgentFlow (dataflow via DSL) / FIDES (runtime), linking the survey, stating our edge as *shipped + imports your real agent + breadth + sound/fail-closed*, explicitly not claiming to have invented the methods. Fixed the stale routing bullet (now rule-based, per 0.11.3) and test count (238). Launch copy drafted in `docs/LAUNCH.md` (HN Show HN, Product Hunt, LinkedIn) — attribution-honest (Cyera 2026 cited, not asserted). GIF deferred to after 0018.
