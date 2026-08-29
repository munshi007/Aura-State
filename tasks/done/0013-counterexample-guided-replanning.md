# 0013: Counterexample-guided replanning (PAT-Agent / VERIMAP)

**Status:** done (2026-08-24)
**Type:** feature
**Tags:** `[verification]` `[core]` `[frontier]`
**Priority:** now (closes the CTL/Z3 loop; build after 0014)
**Depends on:** 0005 (correct CTL + real counterexamples), 0002 (real Z3 counterexamples)
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** core-hardening research (2026-06-01), frontier survey. Refs: PAT-Agent arXiv:2509.23675; VERIMAP arXiv:2510.17109.

## Why

Today verification is a gate: it says VIOLATED and stops. The frontier move is to
**close the loop** — feed the model checker's / solver's counterexample back to
the planner so the agent *repairs* the plan and re-verifies, iterating until the
property holds or a budget is hit. This turns formal verification from a tripwire
into an active controller, and it is the most defensible "verified agent"
narrative: the plan is provably correct *because* the verifier drove it there.

## What

- When CTL (0005) or Z3 (0002) returns a counterexample (a violating path / a
  field assignment), translate it into a structured repair signal and re-invoke
  the planner/replanner with that constraint.
- Loop: plan -> verify -> (if violated) counterexample -> replan -> verify, with a
  bounded iteration count and convergence/abort criteria.
- Integrate with VerificationLoop / ReflectionMemory already in the engine.

## Design

- Counterexample adapter: CTL violating path -> "avoid this transition / ensure X
  before Y"; Z3 model -> "the assignment that breaks the obligation."
- Replanner consumes the constraint (prompt augmentation or hard graph edit) and
  proposes a revised plan/graph.
- Re-verify; stop on PROVEN or after K iterations (then surface unresolved
  violation — no silent accept).
- Persist each iteration in the durable journal (0009) and traces (0010).

## Test strategy

- a graph that violates an ordering property is repaired within K iterations and re-verifies PROVEN
- a genuinely unsatisfiable requirement aborts after K with a clear unresolved-violation result (not a silent pass)
- counterexample adapter maps a known violating path to the expected repair constraint

`test_replan_converges_fixes_0013`, `test_replan_aborts_fixes_0013`.

## Acceptance criteria

- [ ] counterexamples from CTL (0005) and Z3 (0002) translated into structured repair signals
- [ ] plan->verify->replan loop with bounded iterations; converges to PROVEN on the repairable case
- [ ] unsatisfiable case aborts with an explicit unresolved-violation result (no silent accept)
- [ ] iterations persisted in journal + traces
- [ ] docstrings cite PAT-Agent (arXiv:2509.23675) and VERIMAP (arXiv:2510.17109)
- [ ] regression tests `*_fixes_0013` passing

## Notes

_record counterexample-adapter mapping, iteration budget K, and convergence results here_

## Completion (2026-08-24)
`aura_state/core/replan.py`:
- Adapters: `ctl_to_repair` (VerificationResult → target + violating states; parses EF/AF target), `z3_to_repair` (ProofResult → failed obligations + counterexample), `taint_to_repair` (TaintViolation → source/sink/path).
- `counterexample_guided_repair(engine, repair_fn, properties=, check_taint=, max_iterations=)`: verify → first violation (CTL then taint) → RepairSignal → repair_fn → re-verify; converges to PROVEN, or aborts after K with explicit `unresolved` (no silent pass). History on `engine._replan_history`.
- Pluggable `repair_fn` (LLM planner or built-ins). Deterministic built-ins: `insert_sanitizer_repair` (taint → rewire prev→San→sink), `add_edge_to_reach_repair` (CTL reachability → add entry→target edge), `default_repair` dispatch. Z3 declines deterministically (needs a real replanner).
- Engine: `engine.repair(...)`.
Tests `tests/test_replan_fixes_0013.py` (4): taint converges (sanitizer inserted, re-PROVEN), CTL reachability converges (edge added), unrepairable aborts with explicit unresolved, adapters map correctly. Demo `examples/replan_demo.py`. Full suite 117 passed.

Open: LLM-driven replanner for Z3/semantic repairs; persist iterations in the durable journal (0009, deferred) — currently surfaced on `engine._replan_history` + ReplanResult.history.
