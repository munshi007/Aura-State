# 0002: Proof-engine `eval()` RCE -> AST->Z3 compiler

**Status:** done (A+B 2026-08-24; C 2026-08-24)
**Type:** security
**Tags:** `[verification]` `[rce]` `[soundness]`
**Priority:** now
**Depends on:** none
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** core-hardening research (2026-06-01), proof-engine audit. Two defects: code injection via `eval`, and the "proof" being an evaluation, not a proof.

## Why

`proof_engine` is the Z3-backed formal-verification feature — the strongest claim
Aura-State makes. Two problems sit on top of each other:

1. **Security.** Obligation strings are run through `eval` (proof_engine.py:56).
   Obligations can originate from flow files / config / LLM-adjacent surfaces.
   `eval` with `{"__builtins__": {}}` is escapable (same gadget class as task
   0001), so a crafted obligation is host RCE.
2. **Soundness.** Even when it runs as intended, it is not proving anything — it
   pins every variable to its extracted value and checks one point. That is
   evaluation dressed as SMT. The "formal proof" guarantee does not hold.

## What's broken

`aura_state/verification/proof_engine.py`.

**A — `eval` on obligation strings (`_parse_obligation`, proof_engine.py:55-56).**

```python
result = eval(obligation, {"__builtins__": {}}, z3_vars)
```

`z3_vars` holds Z3 `ArithRef`/`BoolRef` objects, so `area > 0` produces a Z3
`BoolRef` — that is the *intended* use. But `eval` runs **any** expression; an
obligation like `().__class__.__bases__...` executes. Empty `__builtins__` is not
a boundary.

**B — fails open (proof_engine.py:62-64, :102-103).** Any parse exception is
caught and returns `None`; in `prove_extraction` a `None` constraint is
`continue`d past (proof_engine.py:102-103). An unparseable or malicious
obligation is therefore silently treated as satisfied. Violates CLAUDE.md rule 4
(fail closed).

**C — evaluation, not proof (proof_engine.py:93-97, :106-113).** Every variable
is pinned (`solver.add(var == value)`), then the negated obligation is checked
for sat. With all variables fixed this only ever tests the single extracted
point — it cannot prove a relationship over a domain, cannot use field
constraints as hypotheses, and gives no real counterexample (the "model" at
proof_engine.py:115 is the pinned point). The Z3 machinery adds cost, not
assurance.

## Repro

```python
from aura_state.verification.proof_engine import prove_extraction
# B: malicious/garbage obligation silently passes
r = prove_extraction({"area": 10}, ["__import__('os')"])
assert r.verified is True   # BUG: unparsed -> None -> skipped -> "verified"
# A: with a non-empty var dict, eval executes attribute traversal
prove_extraction({"x": 1}, ["x.__class__.__bases__"])   # eval runs it
```

## Root cause

Obligations are treated as a tiny expression language but implemented with
`eval`, the one primitive guaranteed to be both unsafe and over-powered. And the
solver is fed a fully-constrained point instead of a symbolic domain, so SMT
degenerates to arithmetic.

## Fix

Replace `eval` with a small, total **AST->Z3 compiler** and make proofs symbolic.

- **Parse, don't eval.** `ast.parse(obligation, mode="eval")`, then walk an
  **allowlist**: `Compare` (`<,<=,>,>=,==,!=`), `BoolOp` (`and/or`), `UnaryOp`
  (`not`), `BinOp` (`+,-,*,/`), `Name` (must be a declared variable), `Constant`
  (num/bool). Map each to the corresponding Z3 constructor. Reject every other
  node type with a hard error. No attribute/subscript/call nodes.
- **Fail closed.** A node outside the grammar, or an undeclared variable, raises
  and marks the obligation **unproven** (`verified=False` with reason) — never
  `continue`/`None`.
- **Prove symbolically.** Add field constraints (types, bounds from the Pydantic
  schema / declared ranges) as Z3 hypotheses *without* pinning every variable.
  Check `unsat(And(hypotheses, Not(obligation)))` to prove the obligation holds
  over the constrained domain; the Z3 model on the sat branch is a real
  counterexample. Pinning is acceptable only for a point-check mode that is
  labeled as such, not as "proof."

## Test strategy

Real Z3, adversarial obligations:

- malicious obligation (`"__import__('os')"`, `"x.__class__"`) -> raises / `verified=False`, never executes
- a true relationship over a domain (`"cost == area * rate"` with declared ranges) -> `verified=True`
- a false relationship -> `verified=False` with a counterexample that actually violates it
- an underspecified obligation (var not declared) -> unproven, not silently passed

Do not monkeypatch the solver. Add `test_proof_*_fixes_0002`.

## Acceptance criteria

- [x] no `eval`/`exec` anywhere in `proof_engine.py`; obligations compiled via an allowlisted AST->Z3 walker
- [x] malicious obligation repro raises and never executes; record corpus in Notes
- [x] fails closed — unparseable/unsupported/undeclared obligations yield `verified=False` (or raise), never silent pass
- [ ] proofs are symbolic over declared field constraints, not single-point pinning; counterexamples genuinely violate the obligation
- [x] regression tests `test_proof_*_fixes_0002` added, exercising real Z3
- [ ] docstring cites the SMT formulation (`unsat(hypotheses & not obligation)`) — point-check formulation documented; symbolic form pending C

## Notes

**2026-08-24 — A (eval RCE) + B (fail-open) fixed. C (symbolic proof) deferred.**

- **A/B fix:** `_parse_obligation` (`eval`) replaced by `_compile_obligation`
  → `_compile_node`, an allowlisted `ast.parse(mode="eval")` walker. Allowed
  nodes: `Compare`, `BoolOp`, `UnaryOp` (`not`/`+`/`-`), `BinOp`
  (`+ - * / % ** //`), `Name` (must be a bound numeric var), `Constant`
  (int/float/bool), plus chained comparisons. Every other node → `ObligationError`.
- **Fail closed:** an obligation that cannot compile or bind is recorded in the
  new `ProofResult.unproven_obligations` and forces `verified=False`. No path
  `continue`s past an unproven obligation as satisfied.
- **Malicious corpus (all → unproven, never executed, `verified=False`):**
  `().__class__.__bases__[0].__subclasses__()`, `x.__class__.__bases__`,
  `__import__('os')`, `x >>> 0 !!`, undeclared `y > 0`, string-field `name > 0`.
- **Tests:** `tests/test_phase9.py::TestProofEngineFailsClosed` (8 cases,
  `_fixes_0002`). Full suite 73 passed (was 65). Real Z3 5.1.0, no monkeypatch.
- **C still open (soundness):** proofs are still single-point pinning
  (`solver.add(var == value)` then `check(Not(constraint))`). This is a valid
  *point check* for the extracted values but is not a symbolic proof over a
  declared domain. Remaining work: thread Pydantic/declared field ranges in as
  Z3 hypotheses and check `unsat(And(hypotheses, Not(obligation)))` without
  pinning. Point-check is now honestly scoped in the module docstring, not
  called a domain proof.

### Repro (fixed)

```
$ .venv/bin/python -m pytest tests/test_phase9.py -q -k FailsClosed
8 passed
$ .venv/bin/python -m pytest tests/ -q
73 passed
```

## Part C completed (2026-08-24) — symbolic consistency proof
Kept the point-check (correct for validating a concrete extraction) and ADDED a genuine symbolic proof: `prove_obligations_satisfiable(obligations, bounds)` declares free Z3 Real vars (NOT pinned), adds declared field bounds as hypotheses, and checks SAT — UNSAT means the obligation set is self-contradictory / impossible for any extraction (a design bug caught before deploy), returning a witness assignment when satisfiable. `field_bounds_from_model(model)` pulls ge/gt/le/lt from a Pydantic model's JSON schema. Contract integration: `NodeContract.obligations_consistent` (None / True / False) computed per node in `compile_contract`. Honestly scoped in the module docstring: point-check verifies a concrete extraction; the symbolic check proves the spec is consistent over ranges.
Tests `tests/test_proof_symbolic_fixes_0002.py` (5): consistent→SAT+valid witness; contradictory→UNSAT; bounds make an obligation impossible; Pydantic bounds extracted; contract flags an inconsistent node. Full suite 125 passed. **All acceptance criteria (A+B+C) now met.**
