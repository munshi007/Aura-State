# 0006: "MCTS" is one-shot UCB1 -> Thompson + CTL-feasibility filter

**Status:** done (2026-08-24)
**Type:** correctness
**Tags:** `[core]` `[routing]` `[rl]`
**Priority:** now
**Depends on:** 0005 (needs CTL feasibility to filter transitions)
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** core-hardening research (2026-06-01), router audit.

## Why

The "MCTS fallback" is marketed as Monte-Carlo Tree Search routing and ties into
the reinforcement-learning story (routes improve from outcomes). It is neither
MCTS nor sound: it is a one-shot UCB1 bandit step with a broken reward scale and
no feasibility constraint, so it can route to CTL-invalid transitions and its
exploration term is meaningless. Misnamed and mis-scaled.

## What's broken

`aura_state/core/engine.py`, `_mcts_select` (~engine.py:224) + `NodeHealthMetrics`
in `aura_state/core/adaptive_graph.py`.

- **Not MCTS.** No tree, no rollouts, no backprop over simulated trajectories —
  a single UCB1 arm selection. Calling it MCTS oversells it.
- **Broken reward scale.** UCB1 assumes rewards in `[0,1]`; the exploration
  constant only makes sense at that scale. The reward fed in is not normalized,
  so the explore/exploit balance is arbitrary.
- **No feasibility filter.** The selector can pick a transition the CTL layer
  (task 0005) would reject — routing and verification disagree.
- **Health metrics too thin.** `NodeHealthMetrics` tracks only
  `total_executions` / `failures` — no per-edge stats, no timestamps, so there is
  no non-stationarity handling and no posterior to sample.

## Repro

```python
# Construct an engine where one transition is CTL-infeasible (e.g. violates an
# ordering/mutual-exclusion property). _mcts_select can still choose it, because
# selection never consults the verifier. And with un-normalized reward the UCB
# bonus dominates or vanishes depending on raw reward magnitude.
```

## Root cause

A bandit step was labeled MCTS, fed an unbounded reward, and wired independently
of the verification layer. The RL framing was aspirational; the implementation is
a single-step heuristic.

## Fix

- **Hard feasibility filter first.** Restrict the candidate set to transitions the
  CTL layer (0005) deems valid from the current state. Routing must never propose
  a verification-invalid move.
- **Real posterior — Thompson / Beta-Bernoulli.** Model each arm's success as
  Beta(alpha, beta); seed priors from prior/expected success as pseudo-counts;
  sample to select. This is well-defined without a reward-scale constant and
  naturally balances explore/exploit.
- **Non-stationarity.** Apply a discount/decay to counts so stale outcomes fade
  (workflows drift). Store per-edge counts + last-update timestamp in
  `NodeHealthMetrics`.
- **Honest naming.** Rename to reflect what it is (bandit router / Thompson
  router). If true MCTS is wanted later, that is a separate task with rollouts +
  backprop; do not keep the MCTS label on a bandit.
- **Bounded reward.** If a scalar reward is retained anywhere, normalize to
  `[0,1]` and document the mapping.

## Test strategy

- feasibility: a CTL-invalid transition is never selectable (real verifier, post-0005)
- convergence: with a clearly-best arm, Thompson selection concentrates on it over rounds (seeded RNG for determinism)
- non-stationarity: after the best arm starts failing, selection shifts within a bounded number of rounds (discounting works)
- no dependence on raw reward magnitude (scale-invariance sanity)

Add `test_router_feasibility_fixes_0006`, `test_router_thompson_fixes_0006`.

## Acceptance criteria

- [ ] candidate transitions filtered to CTL-feasible set before scoring; infeasible transition never selected
- [ ] selection uses a Beta-Bernoulli/Thompson posterior with prior pseudo-counts, not one-shot UCB1
- [ ] per-edge counts + timestamps in `NodeHealthMetrics`; discounting handles non-stationarity
- [ ] component renamed away from "MCTS" unless real tree search is implemented; docstring states the algorithm
- [ ] any retained scalar reward normalized to `[0,1]` with documented mapping
- [ ] regression tests `test_router_*_fixes_0006` added, with seeded RNG and the real verifier

## Notes

_record prior choice, discount factor, rename, and convergence numbers here_

## Completion (2026-08-24)
`_mcts_select` → `_route_select`: honest **Thompson-sampling contextual bandit** (not MCTS). Per-edge Beta-Bernoulli posterior in `adaptive_graph.EdgeStats` (uniform Beta(1,1) prior, discount=0.98 on the surplus each update for non-stationarity, `last_update` timestamp). Success is Bernoulli ∈ [0,1] so no exploration constant / reward scaling. Feasibility filter first: `_is_feasible` requires a structural edge AND (when wired via `set_feasibility_filter`) CTL approval (task 0005 integration point) — infeasible edges never selectable; all-infeasible → END. Seedable `route_seed` for determinism. Tests: `tests/test_router_fixes_0006.py` (4: feasibility, convergence >180/200, non-stationarity shift, all-infeasible→END). Full suite 105 passed.
