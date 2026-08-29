# Tasks

Work backlog for hardening the Aura-State core. Every code change is tied to a
task file here. Source: deep core-hardening research (2026-06-01), a 6-agent
subsystem audit of security, verification correctness, routing, runtime, and
frontier technique.

## Lifecycle (folder kanban)

A task is a single file. It lives in **one stage folder at a time** and is
**moved forward** through the pipeline as work progresses:

```
tasks/backlog/  ->  tasks/scope/  ->  tasks/prototype/  ->  tasks/implementation/  ->  tasks/done/
```

- **backlog/** — captured, not started. Why/What written; blast radius not yet confirmed.
- **scope/** — blast radius confirmed: exact files/functions, repro reproduced, approach chosen. Live audit done where relevant.
- **prototype/** — approach proven on the smallest end-to-end slice. `Prototype review:` set.
- **implementation/** — full build against the scoped surface; tests written.
- **done/** — all acceptance criteria checked, repro no longer reproduces, command+result recorded in Notes.

The folder is the source of truth for stage. The `Status:` field inside each
file is kept in sync as a secondary signal (`git mv` the file **and** update the
`Status:` line in the same commit).

Do not skip `scope/` or `prototype/`. A hand-enumerated function list can miss
drift — the end-to-end repro is the completeness check, not the function list.

> Links below point to a task's **current** folder. When a task moves, update its
> link path here in the same commit as the `git mv`.

## File format

Header fields: `Status`, `Type`, `Tags`, `Priority`, `Depends on`, `Owner`,
`Reviewer`, `Prototype review`, `Found in`. Body sections: `Why`, `What's
broken` (bugs) or `What` (features), `Repro`, `Root cause`, `Fix`, `Test
strategy`, `Acceptance criteria` (checkboxes), `Notes`.

## Backlog

### P0 — security (RCE). Block everything else.
| ID | Title | Type | Priority |
|----|-------|------|----------|
| [0001](backlog/0001-sandbox-rce-subclasses-gadget.md) | Sandbox RCE — `exec` escape via `__subclasses__` gadget | security | now |
| [0002](backlog/0002-proof-engine-eval-rce.md) | Proof-engine `eval()` RCE -> AST->Z3 compiler | security | now |
| [0003](backlog/0003-pickle-rce-tracer-cache.md) | Pickle RCE in tracer/cache -> msgpack + atomic writes | security | now |

### P0 — correctness (flagship guarantees do not hold).
| ID | Title | Type | Priority |
|----|-------|------|----------|
| [0004](backlog/0004-conformal-statistically-invalid.md) | Conformal statistically invalid -> split + order-statistic | correctness | now |
| [0005](backlog/0005-ctl-no-dead-ends-reversed.md) | CTL `no_dead_ends` reversed + deadlock + init-state checks | correctness | now |
| [0006](backlog/0006-mcts-bandit-broken-reward.md) | "MCTS" is one-shot UCB1 -> Thompson + CTL-feasibility filter | correctness | now |
| [0007](backlog/0007-core-bug-sweep.md) | Bug sweep: $0 cost, fake embeddings, logger NameError, dead paths | bug | now |

### P1 — scalability / robustness.
| ID | Title | Type | Priority |
|----|-------|------|----------|
| [0008](backlog/0008-async-core-sync-wrapper.md) | Async core + sync wrapper (AsyncOpenAI + anyio TaskGroup) | infra | later |
| [0009](backlog/0009-durable-journal-statestore.md) | Durable INTENT/RESULT journal + idempotency + `StateStore` ABC | infra | later |
| [0010](backlog/0010-observability-tooling.md) | OpenTelemetry GenAI semconv + lint/typecheck `check.sh` | infra | later |

### P2 — frontier (the differentiation).
| ID | Title | Type | Priority |
|----|-------|------|----------|
| [0011](backlog/0011-pasc-pipeline-aware-conformal.md) | PASC — pipeline-aware conformal | feature | later |
| [0012](backlog/0012-conformal-risk-control-abstention.md) | Conformal Risk Control / Learn-Then-Test abstention | feature | later |
| [0013](backlog/0013-counterexample-guided-replanning.md) | Counterexample-guided replanning (PAT-Agent / VERIMAP) | feature | later |
| [0014](backlog/0014-capability-typed-dataflow.md) | Capability-typed dataflow / taint (prompt-injection proof) | feature | later |

## Dependency order

```
0001 0002 0003   (security, parallel, no deps)
0004 0005 0006   (correctness, parallel; 0006 depends on 0005's feasibility data)
0007             (bug sweep, parallel)
        |
        v
0008 -> 0009 -> 0010   (runtime; 0009 depends on 0008's async boundary)
        |
        v
0011 0012 (depend on 0004)   0013 (depends on 0005)   0014 (depends on 0001+0002 type system)
```

0014 is the "novel -> exciting" unlock; it depends on the security + proof
foundations landing first.
