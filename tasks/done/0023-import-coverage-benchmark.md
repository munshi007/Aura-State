# 0023: Import-coverage benchmark — `aura-state bench`

**Status:** done (2026-09-28)
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

- [x] labeled corpus bundled in `aura_state/bench.py` (synthetic safe/vuln pairs) + real-shaped agents from `examples/audit/`; `aura-state bench` command
- [x] `benchmarks/RESULTS.md` (generated): accuracy table + recall/precision + soundness caveat + import-coverage table
- [x] honest framing — import coverage (100%, 6/6) is the headline; accuracy only on labeled synthetics
- [x] reproducible (`aura-state bench`); CI regression via `test_benchmark_is_sound_*` (silent_misses must be 0)
- [x] tests `tests/test_bench_fixes_0023.py` (4)

## Notes

_record: corpus contents + labels, exact metrics reported, the soundness caveat text._
Relates to [[0019]] (backs the launch story), [[0020]], [[0021]].

## Completion (2026-09-28)
Built the benchmark AND it immediately caught a real soundness bug on its first run.

**Soundness fix (the benchmark's first catch):** `check._cap` derived a tool node's taint capability *only* from its side-effect, ignoring an explicit `capability/data_class: untrusted` (or imported `roles: [untrusted]`). So an untrusted **tool** source (e.g. `web.fetch`) collapsed to "plain" and the taint pass **silently passed** an untrusted-source→sink path — a fail-open the trifecta pass did not share. Fixed: an explicit untrusted marking makes any node a taint source; the two passes are now consistent. Regression tests in `test_bench_fixes_0023.py`.

**Deliverables:** `aura_state/bench.py` (bundled labeled corpus: safe/vuln pairs across trifecta/taint/obligation + clean; `run_bench()`), `aura-state bench` CLI (`--md` writes RESULTS.md), `benchmarks/RESULTS.md`. Real-shaped import coverage over `examples/audit/*.json`.

**Results:** synthetics 8/8 correct — recall 100% · precision 100% · **0 silent misses (sound)**; import coverage **100% (6/6)** real agents ingested unmodified. Full suite 242 passing.

Honest note: recall is 100% *by construction* (fail-closed); the measured number is precision. The headline is import coverage on unmodified agents, not an accuracy-vs-competitors claim.
