# 0015: Design→spec compiler — emit the runtime contract from the verified graph

**Status:** done (2026-08-24)
**Type:** feature
**Tags:** `[core]` `[verification]` `[differentiator]` `[frontier]`
**Priority:** now (next milestone)
**Depends on:** 0005 (CTL over the graph), 0002 (Z3 obligations), 0004 (conformal) — all done
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** strategy synthesis (2026-08-24). Research thesis: the wall in verified agents is the *specification*, not the verifier. NL→policy is only 24–35% faithful; CaMeL / VeriGuard / AgentSpec all work but all name the same limit — "someone must write and maintain the policy." Refs: survey arXiv:2608.14590; AgentSpec (ICSE'26); CaMeL (DeepMind 2025).

## Why

This is the feature that makes Aura-State a framework **no one has built**, and it
sits entirely inside its space (design time), not aura-runtime's (runtime).

Every runtime-assurance system (aura-runtime included) *consumes* a behavioral
contract it cannot author. The spec is the bottleneck: hand-written policy drifts
from the implementation and is only 24–35% faithful when translated from NL.

Aura-State already holds the pieces nobody combines: a typed node graph, Z3
obligations per node, CTL properties over the graph, conformal confidence. If we
**compile a runtime contract from that same design**, the spec is faithful *by
construction* — spec and implementation are one artifact and cannot drift. That
kills both the faithfulness wall and the maintenance burden the whole field
concedes.

Output is the clean seam to aura-runtime later: Aura-State emits the contract,
aura-runtime enforces it. But 0015 is valuable standalone (a design-time
"prove-then-freeze" artifact + diff/regression gate) even before that bridge.

## What

Add a compiler that turns a registered `AuraEngine` (nodes + transitions +
obligations + declared CTL properties) into a single **portable, versioned
contract artifact** — plus a loader/validator and a design-time faithfulness
check (0016 deepens the latter).

The contract must capture, per the current design:
- **Structure**: nodes, declared transitions, entry node, terminals.
- **Data obligations**: each node's Z3 `obligations` (the exact strings + the
  variables they bind), so a runtime monitor can re-check them on live data.
- **Temporal properties**: the CTL properties the graph was proven against
  (reachability / completion / ordering), with their PROVEN/VIOLATED verdict at
  emit time recorded as evidence.
- **Uncertainty policy**: per-node `confidence` (nominal conformal coverage).
- **Provenance**: engine/graph hash, emit timestamp (passed in, not clock-read —
  scripts can't read the clock), Aura-State version, so the artifact is
  content-addressable and diffable.

## Approach

- New module `aura_state/compiler/spec_compiler.py`:
  - `compile_contract(engine, *, properties=None, meta=None) -> AuraContract`
    (a pydantic model). Pulls structure from `engine._transitions` / `_nodes`,
    obligations from each `Node.obligations`, confidence from `Node.confidence`.
    If `properties` given, runs `engine.verify(properties)` and records verdicts.
  - `AuraContract.to_json()` / `from_json()` with **schema-versioned**,
    validated load (reject unknown top-level shape — mirror the tracer's
    fail-closed load).
  - `diff_contracts(a, b)` → structural/obligation/property delta, for a CI
    regression gate ("the contract changed" exits non-zero).
- **Contract format decision (record in Notes):** emit a *native* `AuraContract`
  JSON as the source of truth, plus an **LTLf/AuraSpec-compatible projection**
  so aura-runtime can consume it directly. Do NOT hard-couple to aura-runtime's
  schema in core; keep the projection a separate adapter so Aura-State stays
  standalone. (Confirm aura-runtime's AuraSpec shape before finalizing the
  projection — that repo's `policy.py` is the reference.)
- Wire a CLI/entrypoint: `engine.emit_contract(path)` convenience + a script in
  `examples/` that emits the real-estate pipeline's contract and prints it.
- Faithfulness (0016 owns the depth): here, minimally assert the emitted
  obligations, when re-evaluated against a known-good extraction, reproduce the
  in-loop verdict — i.e. the contract and the loop agree on the same inputs.

## Test strategy

Real objects, no mocks of the compiler:
- build a small engine (2–3 nodes, obligations, one CTL property), compile,
  assert the contract round-trips (`from_json(to_json()) == contract`).
- assert every node obligation + confidence + the transition structure appear in
  the artifact; assert the CTL verdict is recorded.
- **faithfulness**: a good extraction that passes the in-loop verified run must
  also pass when its obligations are re-checked straight from the contract; a
  bad one must fail both. (Contract ≡ loop on the same inputs.)
- `diff_contracts`: change one obligation → non-empty diff; identical graph →
  empty diff.
- malformed/legacy contract JSON → rejected on load, never silently accepted.
Add `test_spec_compiler_*_fixes_0015`.

## Acceptance criteria

- [ ] `compile_contract(engine)` emits a versioned, content-addressable
      `AuraContract` capturing structure + obligations + CTL verdicts + confidence
- [ ] round-trips through JSON with fail-closed, schema-validated load
- [ ] faithfulness check: contract obligations reproduce the in-loop verdict on
      the same inputs (good passes both, bad fails both)
- [ ] `diff_contracts` powers a design-time regression gate
- [ ] an `examples/` script emits and prints the real-estate pipeline's contract
- [ ] LTLf/AuraSpec projection exists as a *separate adapter* (no core coupling),
      with aura-runtime's `policy.py` confirmed as the target shape
- [ ] tests `test_spec_compiler_*_fixes_0015`, real objects, all passing

## Notes

_record: contract JSON schema, the native-vs-AuraSpec decision, aura-runtime
AuraSpec shape as confirmed, and the faithfulness definition used here vs 0016._

Follow-ons: **0016** spec-faithfulness checker (metamorphic/differential — prove
the contract accepts *exactly* intended behavior; ref: executable-spec evaluator,
no LLM judge). Relates to [[0014]] (capability-typed dataflow compiled into the
contract).

## Completion (2026-08-24)
Shipped `aura_state/compiler/spec_compiler.py`: `AuraContract` (versioned, content-addressable, fail-closed `from_json`), `compile_contract(engine, properties=, meta=)` (structure + per-node Z3 obligations + confidence + CTL verdicts as evidence), `check_faithfulness` (contract obligations reproduce the in-loop verdict), `diff_contracts` (regression gate). Engine convenience: `engine.compile_contract()` / `engine.emit_contract(path)`. Exported from top-level. Example: `examples/emit_contract_demo.py` emits + prints the contract and shows good-accepted/bad-rejected. Tests: `tests/test_spec_compiler_fixes_0015.py` (5). Full suite 105 passed.

**Native contract is the source of truth; LTLf/AuraSpec projection deferred to a separate adapter** (not built — to be confirmed against aura-runtime `policy.py` when the bridge is built). Deeper faithfulness (metamorphic/differential) is follow-on 0016.
