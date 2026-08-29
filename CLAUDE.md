# CLAUDE.md

## Project Identity

**Aura-State**: a formally-verified LLM state-machine compiler with calibrated
uncertainty. Python package `aura_state`. Workflows are typed DAGs of `Node`s;
the engine drives an 8-stage `process()` pipeline and the verification layer
issues *guarantees* (Z3 proofs, CTL model checks, conformal intervals) about
what the LLM produced.

The pitch is "agents you can prove things about." That pitch is only as good as
the guarantees actually holding — so **the verification layer is the product**,
and a verifier that fails *open* (passes on error) or claims a guarantee it does
not deliver is worse than no verifier. Treat every `proof`/`conformal`/`CTL`
claim as load-bearing.

## Architecture

```
aura_state/
  core/         engine (AuraEngine, Node, process() pipeline, MCTS/bandit router),
                adaptive_graph (NodeHealthMetrics), verification_loop, providers, exceptions
  verification/ proof_engine (Z3), conformal (split conformal), temporal_verifier (CTL/Kripke)
  execution/    sandbox (English-rule -> Python -> exec), tracer (AuraTrace serialization)
  compiler/     dspy_tuner (few-shot), schema_compiler (JSON-Schema->Pydantic+Node), json_generator
  memory/       pruner (context), trajectory_cache (GraphRAGCache)
  consensus/    auto_vote (majority/unanimous/first_valid)
  loaders/      json_graph (flow.json/yaml -> dynamic Node subclasses)
tests/          test_innovations, test_phase9, test_router
examples/benchmark/   mocked + live real-estate readout
docs/           ALGORITHMS.md, GUIDE.md
```

8-stage `process()`: AdaptiveDAG health -> GraphRAG cache -> Bootstrap few-shot ->
VerificationLoop -> `node.handle()` -> MCTS fallback -> AuraTrace -> speculative exec.

## Stack (non-obvious versions)

- **Python >=3.10**, build backend **hatchling**, deps in `pyproject.toml`.
- **z3-solver >=4.12** — SMT proofs. **pyModelChecking >=1.3** — CTL model checking.
- **instructor >=1.3 + openai >=1.0** — structured LLM extraction.
- **pydantic >=2.0** — extraction schemas. **networkx >=3.0** — DAG. **pyyaml >=6.0** — flow loaders.
- Tests: pytest (+ pytest-asyncio in `[dev]`). No linter/`check.sh` yet (task 0010).

## Critical Rules

1. **No `eval`/`exec` on any string derived from an LLM, an obligation, or a
   flow file.** Parse to an AST and compile to typed ops (Z3 / a whitelisted
   evaluator). `eval` with `{"__builtins__": {}}` is **not** a sandbox — the
   `().__class__.__bases__[0].__subclasses__()` gadget escapes it. Currently
   violated in three places (tasks 0001, 0002) — do not add a fourth.

2. **No `pickle`. Ever.** `pickle.load` on any path reachable from a trace file,
   cache entry, or remote input is RCE. Serialize with msgpack/JSON (task 0003).
   On-disk writes are atomic: temp file -> `os.replace` -> `fsync`.

3. **Sandboxes are deny-by-default allowlists, never blocklists.** Enumerating
   forbidden names (`eval`, `open`, ...) is bypassable by construction. Allowlist
   the exact node types / calls permitted; reject everything else. Prefer no-exec
   evaluation (asteval-style) or real isolation (subprocess+rlimits / WASM) over
   in-process `exec` (task 0001).

4. **Verification fails CLOSED.** A parse error, an unsupported obligation, too
   few calibration samples, or any internal exception means the property is
   **unproven / not covered** — never silently `verified=True` or `continue`d
   past. `proof_engine` currently fails open (`proof_engine.py:64`, `:102-103`).

5. **A guarantee must be the guarantee it claims.**
   - **Conformal** = *split* conformal: calibration set disjoint from the point
     covered; quantile is the k-th order statistic with
     `k = ceil((n+1)*(1-alpha))`, **not** interpolation; minimum
     `n >= ceil(1/alpha) - 1` (=19 at 95%) or it is uncalibrated and must say so.
     Re-running the same input measures *dispersion*, not *error* (task 0004).
   - **CTL** reachability / `AF` must be checked at the **init state**, not over
     all states. **Deadlock detection is structural (BFS), not a CTL formula** —
     totalizing with self-loops (needed for CTL semantics) destroys the
     dead-ends you want to find, so detect them *before* totalizing (task 0005).
   - **Routing** ("MCTS") must filter to CTL-feasible transitions before scoring,
     and use a real posterior (Thompson/Beta-Bernoulli), not one-shot UCB1 with
     a broken reward scale (task 0006).

6. **No fallbacks / silent recovery.** No default values, fallback branches, or
   swallow-and-continue unless explicitly asked. Let errors surface. (The dead
   `instructor` fallback and the `extracted_data=None` speculative path are
   bugs — task 0007.)

7. **Cost and metrics must be real.** `CostTracker` recording 0 tokens (-> $0
   always) and `dspy_tuner` using `ord(char)` as fake embeddings make telemetry
   lie (task 0007).

<important if="you are writing or modifying verification code (proof_engine, conformal, temporal_verifier)">

8. **The test must execute the real solver/checker against an adversarial
   input**, not a mock. Conformal test: assert empirical coverage on a held-out
   set ~ nominal. Proof test: include an obligation that *should* fail and
   confirm `verified=False`. CTL test: include a genuine dead-end / unreachable
   terminal and confirm it is caught. This codebase shipped reversed CTL and
   invalid conformal because tests asserted shape, not correctness.

9. **Cite the math.** When you change a quantile formula, totalization step, or
   bandit update, put the source (paper / order-statistic identity) in the
   docstring and the task file.

</important>

<important if="you are writing or modifying execution code (sandbox, tracer)">

10. **Assume the LLM output is hostile.** Generated code, obligation strings, and
    flow conditions are untrusted input. Before any execution path lands, add a
    test that *attempts the known escape* (`__class__.__bases__`, `__subclasses__`,
    `catch_warnings`, dunder traversal) and asserts it is rejected — task 0001.

11. **No unbounded execution.** Any execution tier enforces CPU time, memory, and
    wall-clock limits, and has no network or filesystem access.

</important>

<important if="you are writing or modifying tests">

12. **Do not monkeypatch the unit under test.** Monkeypatching the LLM boundary
    (network call) is fine; monkeypatching `safe_exec`, the solver, or the
    quantile function hides exactly the drift these tests exist to catch.

13. **Bug-fix tests carry the task id.** Name regression tests
    `test_..._fixes_0NNN`.

</important>

<important if="you are adding or editing a task">

14. **Every code change is tied to a task file under `tasks/`.** Tasks are a
    **folder kanban**: a file lives in one stage dir and is `git mv`'d forward
    through `tasks/backlog/ -> scope/ -> prototype/ -> implementation/ -> done/`.
    The folder is the source of truth; keep the `Status:` line in the file in
    sync in the same commit. Do not jump from `backlog/` straight to
    `implementation/` — `scope/` confirms blast radius (files/functions);
    `prototype/` proves the approach on the smallest end-to-end slice first.

15. **Acceptance criteria are the gate; the repro is the completeness check.**
    Record the exact command + result in the task's Notes before flipping
    `Status: done`. Format / numbering: `tasks/README.md`.

</important>
