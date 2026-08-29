# 0014: Capability-typed dataflow / taint (prompt-injection proof)

**Status:** done (2026-08-24; node-level + value-level; select sub-items intentionally deferred)
**Type:** feature
**Tags:** `[core]` `[security]` `[frontier]` `[differentiator]`
**Priority:** now (differentiator — build after 0015; compiles into the contract)
**Depends on:** 0001 (safe execution), 0002 (typed obligation system), 0005 (graph verification)
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** core-hardening research (2026-06-01), strategic synthesis. Inspiration: CaMeL-style capability/dataflow control for LLM agents.

## Why

This is the "novel -> exciting" unlock. The three existing pillars (Z3, CTL,
conformal) verify graph structure, numeric bounds, and confidence — but **prompt
injection is the #1 reason enterprises won't deploy agents**, and none of those
pillars touch it. Capability-typed dataflow tracks which data is untrusted and
**proves untrusted data cannot reach a dangerous tool/action**. It fits
Aura-State's existing typed-DAG model perfectly and turns the framework from
"impressive math on the wrong target" into "the agent framework that can prove
the property buyers actually lose sleep over." Highest-excitement addition in the
backlog.

## What

- **Capability types on data.** Tag values with provenance/trust labels (e.g.
  `Untrusted`, `Trusted`, capability tokens for tool access). Labels propagate
  through node transitions (taint propagation).
- **Dangerous-sink declarations.** Nodes/tools that perform side effects declare
  required capabilities (e.g. "send-email requires Trusted recipient").
- **Static proof of non-flow.** Before execution, prove (via the typed DAG +
  Z3/dataflow analysis) that no untrusted value can reach a sink lacking the
  capability — a provable anti-prompt-injection guarantee. Violations are caught
  at compile/verify time, not runtime.
- Compose with 0001 (the executor can't be escaped) and 0002 (typed constraints).

## Design

- Extend `Node` / the typed-DAG type system with a capability/label lattice.
- A dataflow pass over the graph computes the label of each value reaching each
  sink; reject any flow where an untrusted label reaches a guarded sink.
- Express the non-interference property in the existing verification layer so it
  reports like the other guarantees (PROVEN/VIOLATED with a counterexample path,
  feeding 0013 replanning).
- Document the threat model and the exact guarantee (and its limits — it bounds
  *flow*, not semantic correctness of trusted data).

## Test strategy

- a graph that routes untrusted user text into a guarded tool -> VIOLATED with the offending flow path
- a graph that sanitizes/elevates trust before the sink -> PROVEN
- label propagation correctness across multi-hop transitions
- composition: a proven-safe-flow graph still runs; a violating one is blocked pre-execution

`test_dataflow_injection_blocked_fixes_0014`, `test_dataflow_safe_flow_fixes_0014`.

## Acceptance criteria

- [ ] capability/trust label lattice on data; taint propagates across node transitions
- [ ] sinks declare required capabilities; violations (untrusted -> guarded sink) caught at verify time with a counterexample path
- [ ] non-interference property reported through the existing verification layer; feeds 0013 replanning
- [ ] threat model + exact guarantee + limits documented
- [ ] regression tests `*_fixes_0014` passing (injection blocked, safe flow allowed)
- [ ] composes with 0001 (executor) and 0002 (typed constraints)

## Notes

_record the label lattice, sink-capability declarations, and the documented threat model here_

## Prototype shipped (2026-08-24)
Node-level static taint over the typed graph (Agentproof-style, arXiv:2603.20356), NOT CaMeL's runtime interpreter — design-time, so the verdict compiles into the contract.
- `aura_state/verification/taint.py`: `analyze_taint(engine) -> TaintResult` (sound may-reach DFS/fixpoint; sanitizers prune; violations carry the concrete source→sink path).
- `Node` labels: `untrusted_source`, `dangerous_sink`, `sanitizer`. `engine.analyze_taint()`.
- Contract integration: `AuraContract.taint` (TaintContract: verdict + violation paths); verdict is part of the content hash, so adding a sanitizer changes the contract.
- Tests `tests/test_taint_fixes_0014.py` (4). Demo `examples/taint_proof_demo.py` (VIOLATED→add sanitizer→PROVEN). Full suite 109 passed.

**Still open (full 0014):** value/field-level granularity (per-value provenance, not just per-node); deriving sanitizer status from a node's Z3 obligations rather than an explicit flag; capability policies richer than source/sink/sanitizer (e.g. "recipient must be user-provided"); feed the violating path into 0013 counterexample-guided repair. Follow-on 0016 faithfulness applies here too.

## Value-level completed (2026-08-24)
`analyze_field_taint(engine)` (in `verification/taint.py`) — field-level static taint: a fixpoint over per-field taint state `{field -> origin}` along the graph, transfer `out(n) = introduced(n) ∪ (in(n) \ sanitized(n))`, with `"*"` wildcard for schema-less untrusted sources. Precise: a clean field passes a sink untouched; only a tainted field matching a sink's `sink_fields` is a violation, attributed to the field + origin node + path.
Node fields: `untrusted_fields`, `sink_fields`, `sanitizes_fields` (refine the node-level flags; `untrusted_source` + a Pydantic `extracts` schema ⇒ all extracted fields untrusted by default). `engine.analyze_field_taint()`. Contract now uses field-level (`TaintPath.field`).
Tests `tests/test_field_taint_fixes_0014.py` (5): clean field passes; tainted field caught; field-specific sanitizer clears only its field; schema fields untrusted by default; contract carries the field. Full suite 130 passed.

**Intentionally deferred / not done (documented decision):** deriving sanitizer status from a node's Z3 *obligations* — rejected as unsound (data-validity ≠ injection-safety; a `budget > 0` obligation does not sanitize prompt injection). Richer capability policies (e.g. "recipient must be user-provided") and feeding field-level violations into 0013's repair are optional future work; the node-level taint→repair adapter already exists.
