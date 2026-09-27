# 0023: Import-coverage benchmark — `aura-state bench`

**Status:** backlog
**Type:** feature
**Tags:** `[research]` `[benchmark]` `[launch]`
**Priority:** later (Phase 1 — depth)
**Depends on:** 0020 (IFC), 0021 (capability-containment)
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** competitive verification pass (2026-09-27). The survey (arXiv:2608.14590) confirms all agent-safety benchmarks are **post-hoc/runtime** (AgentHarm, AgentBench, Agent-SafetyBench, PlanBench). The competitors can't ingest an existing agent at all (AgentFlow=DSL, FIDES=planner, AgentProof=topology). So the honest, defensible benchmark angle is **import coverage on real unmodified agents**, not "we detect better."

## Why

We must NOT claim higher detection accuracy than the prototypes (unverified, contested — some are more rigorous). The true, checkable story: **we run on N real GitHub agents unmodified, in <1s, and report the findings; the others require a rewrite to even start.** That's the moat, measured.

## What

- `benchmarks/` corpus: real scanned agents (CrewAI/LangGraph/AutoGen/MCP from the 0.11.2 work) + synthetic safe/vulnerable pairs (trifecta + least-privilege), each with ground-truth labels.
- `aura-state bench` — runs the corpus, prints a `RESULTS.md` table: per-agent verdict, **import success**, findings, wall-time, and (on labeled synthetics) recall/precision reported **honestly** (sound = fail-closed; some false positives; unknowns surfaced).
- Reproducible: anyone can run it.

## Approach

- Corpus + harness under `benchmarks/`; a CLI entrypoint `aura-state bench [--corpus …]`.
- Report import-coverage + findings + time as the headline; recall/precision only on the labeled synthetic split, with the soundness caveat stated in the output.
- CI regression: fail if precision/recall on the synthetic split drops.

## Test strategy

- `test_bench_runs_and_reports_fixes_0023`: bench runs the corpus, emits the table, labels match on the synthetic split.
- Import-coverage counts the real agents that parse to a flow.
- Regression guard: a seeded false-negative in a synthetic agent makes bench fail.

## Acceptance criteria

- [ ] `benchmarks/` corpus (real + labeled synthetic) + `aura-state bench`
- [ ] `RESULTS.md` table: import success, findings, time; recall/precision on synthetics with soundness caveat
- [ ] honest framing — import coverage on unmodified agents is the headline, not accuracy-vs-others
- [ ] reproducible; CI regression on the synthetic split
- [ ] tests `test_bench_*_fixes_0023`

## Notes

_record: corpus contents + labels, exact metrics reported, the soundness caveat text._
Relates to [[0019]] (backs the launch story), [[0020]], [[0021]].
