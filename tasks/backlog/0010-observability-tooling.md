# 0010: OpenTelemetry GenAI semconv + lint/typecheck `check.sh`

**Status:** backlog
**Type:** infra
**Tags:** `[observability]` `[tooling]`
**Priority:** reconsider (2026-08-24 triage: OTel ingest IS aura-runtime's boundary; building it here duplicates the sibling)
**Depends on:** 0008 (async spans), 0009 (durable events)
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** core-hardening research (2026-06-01), runtime/tooling audit.

## Why

There is no standard telemetry and no quality gate. Enterprise adopters expect
OpenTelemetry traces and a repeatable `check.sh` (the reference repo's CLAUDE.md
treats `./tools/check.sh` as the merge gate). Without spans, a multi-node agent
run is a black box; without lint/typecheck, drift like the bugs in 0007 ships
unnoticed.

## What

- **OpenTelemetry GenAI semantic conventions.** Emit spans per node, per LLM call,
  per verification step, using the GenAI semconv attributes (model, token usage,
  cost, verification verdicts). One trace per session; spans nest along the DAG.
- **`tools/check.sh`.** A single gate: lint (ruff), typecheck (mypy, strict where
  feasible), `pytest`. Wire as the documented pre-merge command in CLAUDE.md.
- **CI hook** (optional): run `check.sh` on PRs.

## Design

- A thin tracing layer in the engine `process()` loop and around LLM/verification
  calls; no-op when no exporter configured (zero overhead by default).
- Token usage + cost (from task 0007's fixed `CostTracker`) attached as span
  attributes — telemetry now reflects real numbers.
- `check.sh`: `ruff check . && mypy aura_state && pytest -q`.

## Test strategy

- with an in-memory span exporter, a single `process()` run produces the expected span tree (node + llm + verification spans) with token/cost attributes
- `check.sh` exits non-zero on a deliberately-introduced lint/type/test failure and zero on a clean tree

`test_otel_spans_fixes_0010`.

## Acceptance criteria

- [ ] OTel GenAI-semconv spans for node/LLM/verification steps; no-op without an exporter
- [ ] span attributes carry real token usage + cost (depends on 0007)
- [ ] `tools/check.sh` runs ruff + mypy + pytest; documented in CLAUDE.md as the merge gate
- [ ] span-tree regression test `test_otel_spans_fixes_0010` passing
- [ ] `check.sh` green on the repo

## Notes

_record exporter choice, mypy strictness level, and span schema here_
