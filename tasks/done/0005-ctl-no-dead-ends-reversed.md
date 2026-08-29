# 0005: CTL `no_dead_ends` reversed + deadlock + init-state checks

**Status:** done (2026-08-24)
**Type:** correctness
**Tags:** `[verification]` `[ctl]` `[soundness]`
**Priority:** now
**Depends on:** none
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** core-hardening research (2026-06-01), temporal-verifier audit.

## Why

CTL model checking is the workflow-soundness guarantee ("every path reaches an
end," "X never happens before Y," "no dead-ends"). Three of the built-in
properties return the wrong formula or check at the wrong scope, so the verifier
reports PROVEN/VIOLATED incorrectly. A model checker that lies about graph safety
is worse than none — users trust it precisely because it is "formal."

## What's broken

`aura_state/verification/temporal_verifier.py`.

**A — `no_dead_ends` is reversed (temporal_verifier.py:91-95).**

```python
def no_dead_ends():
    """AG(not terminal) is false if terminals exist, so instead: ... structural."""
    return A(G(Not("terminal")))      # :95  -- returns the thing the docstring says is wrong
```

The docstring acknowledges `AG(¬terminal)` is the wrong check, then returns
exactly that. `AG(¬terminal)` asserts *no state is ever terminal* — it fails on
every well-formed workflow (which must reach a terminal) and "passes" only graphs
that never finish. Inverted.

**B — deadlock conflated with totalization (temporal_verifier.py:55-57).**

```python
if not targets:
    props.add("terminal")
    edges.append((node_name, node_name))    # self-loop
```

Adding self-loops to sink states is required for CTL semantics (the transition
relation must be total). But it is done in the *same* pass that should detect
dead-ends — after this, every sink has a successor, so no CTL formula can find a
genuine dead-end. Deadlock/dead-end detection must be a **structural BFS over the
original (pre-totalized) graph**, separate from the totalization step. An
intended-terminal vs an accidental dead-end are indistinguishable here.

**C — properties checked at all states, not the init state
(`verify_property`, temporal_verifier.py:98-105).**

```python
satisfying = modelcheck(kripke, formula)
violating = all_names - satisfying_names     # :103
result = VIOLATED if violating else PROVEN     # :105
```

For existential/eventual properties (`EF target`, `AF completion`) the question
is "does the **initial** state satisfy it," not "do all states." A reachable
target legitimately is not satisfied *at every* state, so this reports VIOLATED on
correct graphs. Reachability and `AF` must be evaluated at the init/entry state.

**D — `always_before` is label-coincidence (temporal_verifier.py:69-75).**
`A(G(Imply(after, before)))` checks that the `after` and `before` *labels*
co-occur on a state, not that `before` temporally precedes `after` on all paths.
The correct ordering property is `¬E[¬before U (after ∧ ¬before)]` (after cannot
occur on a path until before has held) — needs the `U` operator, not `G(Imply)`.

## Repro

```python
from aura_state.verification import temporal_verifier as tv
# A graph: start -> end (end is the only terminal)
nodes = {"start": N(targets=["end"]), "end": N(targets=[])}
k = tv.compile_kripke(nodes, {"start":["end"]})
# no_dead_ends() returns A(G(Not("terminal"))) -> VIOLATED, though graph is healthy
# eventual_completion("end") -> checked over all states -> "start" doesn't satisfy AF at... reports VIOLATED
# a genuine dead-end (a node with no targets that is NOT an intended terminal) is masked by the self-loop
```

## Root cause

CTL semantics (total transition relation, evaluation at the init state, `U` for
ordering) were approximated with label-level `G(Imply)` tricks and an all-states
sweep. Totalization (a *modeling* requirement) was fused with deadlock detection
(an *analysis* requirement), which are opposites.

## Fix

- **Separate deadlock detection from totalization.** Before adding any self-loop,
  run a structural pass over the original graph: a node with no outgoing edge
  that is **not** declared an intended terminal is a dead-end -> report it.
  Encode availability as `¬EF(deadlock)` only after marking real deadlocks with a
  dedicated atomic prop, or report structurally (BFS) and skip CTL for this.
  Then totalize (self-loops on intended terminals) purely for CTL well-formedness.
- **Fix `no_dead_ends`** to mean what it says: every non-terminal node has a
  successor in the original graph (structural), or `¬EF(deadlock)` over the
  marked structure.
- **Evaluate at the init state.** `verify_property` takes the entry node;
  `EF`/`AF`/reachability are PROVEN iff the init state is in the satisfying set.
  Universal-safety (`AG ...`) may still range over reachable states, but compute
  the **reachable** set, not `all_states`.
- **Fix `always_before`** to the until-based ordering formula
  `¬E[¬before U (after ∧ ¬before)]`.

This task feeds 0006 (the router must filter to CTL-feasible transitions) and
0013 (counterexample replanning).

## Test strategy

Graphs with known answers (CLAUDE.md rule 8), real `pyModelChecking`:

- healthy `start->end`: `no_dead_ends` PROVEN, `eventual_completion("end")` PROVEN
- a true dead-end (`mid` with no targets, not declared terminal): detected, reported
- unreachable terminal: reachability VIOLATED at init
- ordering: `before` truly precedes `after` -> PROVEN; a path with `after` before `before` -> VIOLATED

Add `test_ctl_dead_end_*_fixes_0005`, `test_ctl_init_state_fixes_0005`,
`test_ctl_ordering_fixes_0005`.

## Acceptance criteria

- [ ] deadlock/dead-end detection is structural over the pre-totalized graph and distinguishes intended terminals from accidental sinks
- [ ] `no_dead_ends` returns a formula/structural check whose PROVEN result means "no accidental dead-ends," verified on the healthy-graph repro
- [ ] `EF`/`AF`/reachability evaluated at the init state; `AG` ranges over reachable states, not `all_states`
- [ ] `always_before` uses the until-based ordering formula, not `G(Imply)`
- [ ] regression tests `test_ctl_*_fixes_0005` added, exercising real model checking with known-answer graphs
- [ ] docstrings state the CTL semantics and why totalization is separate from deadlock detection

## Notes

_record the structural-vs-CTL decision for deadlock detection and the init-state plumbing here_

## Completion (2026-08-24)
Deadlock detection made structural (`find_dead_ends` over the pre-totalized graph; sink ∉ declared terminals = dead-end) and separated from totalization (self-loops still added for pyModelChecking's total-relation requirement, but intended terminals get prop `terminal`, accidental sinks `dead_end`). `no_dead_ends` de-reversed → PROVEN when no accidental sinks. `verify_property`/`verify_engine` now evaluate EF/AF/reachability at the auto-detected init state (PROVEN iff init ∈ satisfying); violating states over the reachable set. `always_before` now uses the until formula `¬E[¬before U (after ∧ ¬before)]`. Tests: `tests/test_ctl_fixes_0005.py` (3). Full suite 105 passed.
