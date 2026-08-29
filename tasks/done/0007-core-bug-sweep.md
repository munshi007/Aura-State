# 0007: Bug sweep — $0 cost, fake embeddings, logger NameError, dead paths

**Status:** done (2026-08-24)
**Type:** bug
**Tags:** `[core]` `[memory]` `[compiler]` `[telemetry]`
**Priority:** now
**Depends on:** none
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** core-hardening research (2026-06-01), cross-subsystem audit. Smaller, independent defects that each make the system lie or crash on a common path.

## Why

A cluster of bugs that aren't security or soundness but quietly break trust:
telemetry reports $0 always, the cache crashes on a hit, "embeddings" are
`ord(char)`, and dead branches confuse anyone reading the code. None is large;
together they undermine "production-ready." Batched because each is a small,
isolated fix with its own regression test.

## What's broken

**B1 — `CostTracker` never records -> cost always $0.**
`aura_state/core/providers.py` has `record(...)` (providers.py:~75-85) and
`total_cost_usd` (providers.py:30-33), but `engine.py` only calls
`adaptive_graph.record_execution` (engine.py:321, 330, 398, 421) — it never calls
the cost tracker, and token counts from the LLM response (`response.usage`) are
never threaded in. Result: `input_tokens`/`output_tokens` stay 0, every cost is
$0, budget enforcement is a no-op.

**B2 — `GraphRAGCache` logger NameError on cache hit.**
`aura_state/memory/trajectory_cache.py` calls `logger.info(...)` at
trajectory_cache.py:98 (cache hit) and :116 (store) but never imports or defines
`logger` (top of file imports json/os/networkx/typing/pydantic only). Every cache
hit raises `NameError: name 'logger' is not defined` — the cache path crashes
exactly when it should help.

**B3 — fake `ord(char)` embeddings.**
`aura_state/compiler/dspy_tuner.py:34` builds "embeddings" as `float(ord(char))`
per character (docstring: "Simple character-based embedding for testing. In
production, swap for ..."). Few-shot example selection / similarity over these is
meaningless — it ranks by byte value, not semantics.

**B4 — dead `instructor` fallback path.**
`aura_state/core/engine.py` (prior audit ~engine.py:358) has an `instructor`
fallback branch that is unreachable / never triggers given the single patched
client set at engine.py:88. Confirm exact lines in scope; remove or wire it.

**B5 — speculative execution with `extracted_data=None`.**
`_speculative_process_node` / `_speculative_execute` (engine.py:158-191) run
`handle()` speculatively with no extracted data, which can call a node handler in
a state it never sees in the real path (prior audit ~engine.py:187). Confirm and
either pass real extracted data or restrict speculation to side-effect-free
handler logic (the comment at engine.py:191 claims "only run non-LLM parts" —
verify that holds).

## Repro

```python
# B2: any cache hit
from aura_state.memory.trajectory_cache import GraphRAGCache
# populate + lookup an isomorphic graph -> NameError: name 'logger' is not defined

# B1: run any node through the engine, inspect CostTracker -> total_cost_usd == 0.0
#     despite real token usage in the OpenAI response
```

## Fix

- **B1:** thread `response.usage.prompt_tokens` / `completion_tokens` from each
  LLM call into `CostTracker.record(node, model, input_tokens, output_tokens,
  latency_ms)`; set real `*_cost_per_m` per model; make budget enforcement read
  the populated tracker.
- **B2:** add `logger = logging.getLogger("aura_state.cache")` (and `import
  logging`) to `trajectory_cache.py`.
- **B3:** replace `ord`-embeddings with a real embedding call (OpenAI embeddings
  or a local model) behind the existing seam; if offline determinism is needed
  for tests, gate the stub behind an explicit `embedder` injection, not as the
  default. Do not ship `ord` as the default.
- **B4:** confirm the dead branch; delete it (no fallbacks per CLAUDE.md rule 6)
  or wire it to a real purpose with a test.
- **B5:** confirm what runs speculatively; ensure no node handler executes with
  `extracted_data=None` in a way it would never see live, or restrict speculation
  to pure routing/no-LLM work as the comment claims.

## Test strategy

One regression test per bug, real objects:

- `test_cost_tracker_records_fixes_0007` — after a (mocked-LLM-boundary) call, `total_cost_usd > 0` for non-zero usage
- `test_cache_hit_no_nameerror_fixes_0007` — a cache hit returns without raising
- `test_embeddings_not_ord_fixes_0007` — default embedder is not the `ord` stub; injected stub still works for offline tests
- B4/B5 — a test pinning the corrected behavior (no dead branch; no `None`-data handler call)

Monkeypatch only the network/LLM boundary, never the unit under test.

## Acceptance criteria

- [ ] B1: real token usage threaded into `CostTracker`; `total_cost_usd > 0` for non-zero usage; budget enforcement actually triggers
- [ ] B2: `logger` defined in `trajectory_cache.py`; cache hit + store no longer raise `NameError`
- [ ] B3: default embedding path is a real embedder; `ord` stub only via explicit injection
- [ ] B4: dead `instructor` branch removed or wired with a test (state which)
- [ ] B5: no speculative handler runs with `extracted_data=None` it would never see live; speculation scope documented
- [ ] one regression test per bug (`*_fixes_0007`), all passing
- [ ] exact line numbers for B4/B5 confirmed and recorded in Notes

## Notes

_record confirmed B4/B5 line numbers, model cost table, and embedder choice here_

## Completion (2026-08-24)
- **B1:** real token usage threaded via `create_with_completion` → `CostTracker.record`; benchmark now reports **$0.0130** (was $0.0). `_usage_from_completion` reads prompt/completion tokens, (0,0) only when provider truly reports none (no fabricated default).
- **B2:** `logger = logging.getLogger("aura_state.cache")` added to `trajectory_cache.py`; cache hit/store no longer NameError.
- **B3:** `ord()` embedding default removed. `BootstrapTeleprompter(embedder=… | openai_client=…)`; no embedder → fail-loud `RuntimeError`. `char_stub_embedder` retained for explicit test injection only.
- **B4:** dead `self.client.chat.completions.create` else-branch removed (engine.py ~358); extraction always routes through `provider.extract`.
- **B5:** speculation now skips any node with an `extracts` schema (engine.py `_speculative_execute`) — an extraction-free node genuinely gets `extracted_data=None` live, so speculating it is faithful; extract-nodes are no longer run with fake None data.
Tests: `tests/test_bugsweep_fixes_0007.py` (5). Full suite 105 passed.
