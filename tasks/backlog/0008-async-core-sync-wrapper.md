# 0008: Async core + sync wrapper (AsyncOpenAI + anyio TaskGroup)

**Status:** backlog
**Type:** infra
**Tags:** `[core]` `[scalability]` `[runtime]`
**Priority:** later (only if Aura-State needs throughput; otherwise defer)
**Depends on:** 0007 (clean engine paths first)
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** core-hardening research (2026-06-01), runtime audit.

## Why

The engine is synchronous and parallelizes speculation with a
`ThreadPoolExecutor(max_workers=4)` (engine.py:102). LLM calls are IO-bound;
threads cap concurrency, leak under load, and make cancellation/timeouts ad hoc.
To be "scalable" the core must be async-first: concurrent node fan-out, real
timeouts, backpressure, and clean cancellation. A sync wrapper preserves the
current API.

## What

- **Async core.** Convert the `process()` pipeline and `node.handle()` to
  `async def`; use `AsyncOpenAI` (async instructor) for all LLM calls.
- **Structured concurrency.** Replace the `ThreadPoolExecutor` speculation with an
  `anyio`/`asyncio` TaskGroup: bounded concurrency, propagated cancellation,
  per-task timeout. Speculative branches cancel cleanly when the real path
  resolves.
- **Sync wrapper.** Keep a thin synchronous `process()` that runs the async core
  via `anyio.run` / `asyncio.run` so existing callers and examples don't break.
- **Backpressure.** A concurrency limit (semaphore) so large graphs don't open
  unbounded LLM connections.

## Design

- `AuraEngine.aprocess(...)` is the real implementation; `process(...)` wraps it.
- LLM client becomes `AsyncOpenAI` patched by instructor's async mode; the
  multi-provider router (providers.py) gains async `acomplete`.
- Speculation: `async with anyio.create_task_group() as tg:` spawn candidates,
  cancel the group when the chosen branch returns.
- Timeouts: `with anyio.fail_after(t):` around each LLM call; surface as a typed
  error (no silent fallback — CLAUDE.md rule 6).

## Test strategy

- async path returns identical results to the (pre-change) sync path on the benchmark
- speculative branches cancel when the real branch resolves (assert no orphan tasks, no extra LLM calls)
- a per-call timeout fires and surfaces a typed error, not a hang
- the sync wrapper still works for existing examples

Mock the LLM boundary (async), not the engine. `test_async_*_fixes_0008`.

## Acceptance criteria

- [ ] `aprocess` async core with `AsyncOpenAI`; sync `process` wrapper preserves the current API and benchmark results
- [ ] speculation uses a TaskGroup with bounded concurrency + cancellation; no `ThreadPoolExecutor`
- [ ] per-LLM-call timeout surfaces a typed error (no silent fallback)
- [ ] concurrency semaphore bounds in-flight LLM calls; documented default
- [ ] regression tests `test_async_*_fixes_0008` passing; benchmark parity recorded in Notes

## Notes

_record concurrency default, anyio-vs-asyncio choice, and benchmark parity numbers here_
