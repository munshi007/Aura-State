# 0009: Durable INTENT/RESULT journal + idempotency + `StateStore` ABC

**Status:** backlog
**Type:** infra
**Tags:** `[core]` `[durability]` `[scalability]`
**Priority:** reconsider (2026-08-24 triage: durable journal/evidence is aura-runtime's job now; likely belongs there, not here)
**Depends on:** 0008 (async boundary), 0003 (safe serialization)
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** core-hardening research (2026-06-01), runtime audit.

## Why

State lives in memory; a crash mid-workflow loses progress and can re-run
side-effecting nodes (duplicate LLM spend, duplicate external actions). For
enterprise reliability the engine needs durable execution: a write-ahead journal,
idempotency keys so replays don't double-execute, and a pluggable backend so
state can live in memory (dev), SQLite (single-node), or Postgres/Redis
(distributed).

## What

- **`StateStore` ABC.** Abstract interface: `append_intent`, `record_result`,
  `load_session`, `checkpoint`. Implementations: in-memory (default/dev), SQLite,
  and a documented Postgres/Redis seam.
- **INTENT/RESULT journal.** Before a node executes, append an INTENT (node,
  input hash, idempotency key). After, append a RESULT. On restart, replay the
  journal: completed nodes (RESULT present) are skipped; an INTENT without a
  RESULT is re-attempted under the same idempotency key.
- **Idempotency keys.** Deterministic key per (session, node, input) so a replay
  of a side-effecting node is deduplicated rather than re-run.
- Serialization via msgpack/JSON + atomic writes (reuse task 0003 infra).

## Design

- `process()` wraps each node step: `store.append_intent(...)` ->
  `node.handle()` -> `store.record_result(...)`.
- Replay on `load_session`: build the completed-node set from RESULT records;
  resume from the frontier.
- Idempotency: external/side-effecting handlers receive the key and must no-op on
  a seen key (documented contract).

## Test strategy

- kill the engine after INTENT but before RESULT; restart; assert the node is re-attempted exactly once and downstream completes
- kill after RESULT; restart; assert the node is skipped (not re-run)
- idempotent replay: a side-effecting node invoked twice with the same key executes its effect once
- `StateStore` conformance suite run against in-memory + SQLite

Real store + real files; mock only external side effects. `test_durable_*_fixes_0009`.

## Acceptance criteria

- [ ] `StateStore` ABC with in-memory + SQLite implementations passing a shared conformance suite
- [ ] INTENT/RESULT journal; restart replays correctly (completed skipped, incomplete retried once)
- [ ] idempotency keys dedupe side-effecting replays
- [ ] journal serialized via msgpack/JSON with atomic writes (no pickle)
- [ ] crash-recovery regression tests `test_durable_*_fixes_0009` passing
- [ ] Postgres/Redis seam documented (not necessarily implemented)

## Notes

_record store schema, idempotency-key derivation, and recovery test results here_
