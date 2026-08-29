# 0001: Sandbox RCE — `exec` escape via `__subclasses__` gadget

**Status:** done (2026-08-24)
**Type:** security
**Tags:** `[execution]` `[rce]`
**Priority:** now
**Depends on:** none
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** core-hardening research (2026-06-01), sandbox subsystem audit. Escape reproduced against `SandboxedInterpreter.safe_exec`.

## Why

`SandboxedInterpreter` is sold as the feature that "prevents LLM calculation
hallucinations" by running generated Python in a "restricted sandbox." It is not
restricted. An attacker who controls the English rule (or the LLM output for it)
gets arbitrary code execution in the host process: filesystem, network,
environment secrets, the OpenAI key. This is the single highest-severity defect
in the repo — it converts an LLM prompt-injection into host RCE.

## What's broken

`aura_state/execution/sandbox.py`. Two compounding holes:

**A — blocklist AST validation (`_validate_ast`, sandbox.py:28-39).** It walks the
AST and rejects only:
- `ast.Import` / `ast.ImportFrom` (sandbox.py:33)
- `ast.Call` whose `func` is an `ast.Name` in `{eval, exec, open, __import__, globals, locals}` (sandbox.py:36)

It never inspects attribute access. The classic escape uses no import and no
banned name:

```python
result = ().__class__.__bases__[0].__subclasses__()
# -> walk to <class 'subprocess.Popen'> or warnings.catch_warnings
#    ._module.__builtins__['__import__']('os').system('...')
```

`getattr`, comprehensions, lambdas, f-strings with `{}` calls, and decorator
expressions are likewise unchecked.

**B — in-process `exec` (sandbox.py:55).** Even with a clean AST, `exec(code_str,
restricted_globals, local_vars)` runs in the host interpreter with no CPU,
memory, wall-clock, or fd limits. `{"__builtins__": {...10 funcs}}` is not an
isolation boundary — object traversal reaches the full type graph.

## Repro

```python
from aura_state.execution.sandbox import SandboxedInterpreter
si = SandboxedInterpreter()
payload = (
    "result = ()."
    "__class__.__bases__[0].__subclasses__()"
)
# passes _validate_ast (no import, no banned Name), then exec runs it.
print(si.safe_exec(payload, {}))   # leaks the live subclass list -> gadget chain
```

A full chain reaching `subprocess` / `os.system` runs to completion. No
exception, no log warning.

## Root cause

Security modeled as a *blocklist of names* over a language whose object model is
fully reachable by attribute traversal. A blocklist cannot enumerate the escape
surface of `exec`. The only correct postures are (a) never `exec` — evaluate a
restricted grammar with no attribute/dunder access, or (b) execute under real OS
isolation.

## Fix

Deny-by-default. Pick the tier that matches the need; recommended path is tier 1
for the math/logic use case actually present, with tier 2 as the escape hatch.

- **Tier 1 (recommended default) — no `exec`, allowlist evaluator.** Replace
  `safe_exec` with an `asteval`-style interpreter (or a hand-rolled AST walker)
  that **allowlists** node types: numbers, names bound to provided inputs,
  `BinOp`, `UnaryOp`, `Compare`, `BoolOp`, and a fixed set of safe calls
  (`abs/min/max/sum/round/int/float/bool/len`). **Reject** `Attribute`,
  `Subscript` on arbitrary objects, `Lambda`, comprehensions, dunder names
  (anything matching `__\w+__`), and every node not on the allowlist. This covers
  the real workload (arithmetic / logic over extracted fields) with zero exec.
- **Tier 2 — real isolation for genuine code.** If arbitrary Python is ever
  required, run in `subprocess` with `resource.setrlimit` (RLIMIT_CPU,
  RLIMIT_AS), a wall-clock timeout, `PYTHONPATH` stripped, no network namespace,
  and a read-only temp cwd. Better: Pyodide/WASM or an E2B/Firecracker microVM.
- Forbid attribute access containing `__` regardless of tier.
- Surface rejections as `SandboxExecutionError` with the offending node type.

## Test strategy

A test that runs the **known escapes** and asserts each raises
`SandboxExecutionError` — not a mock. Minimum corpus:

- `().__class__.__bases__[0].__subclasses__()`
- `getattr((), '__class__')`
- `[x for x in ().__class__.__mro__]`
- `(lambda: ().__class__)()`
- `__import__('os')` (already caught — keep as regression)
- a benign arithmetic rule that **must still succeed** (`result = max(a, b) * 1.2`).

Per CLAUDE.md rule 12: do not monkeypatch `safe_exec`. The test calls the real
evaluator. Add `test_sandbox_escape_blocked_fixes_0001`.

## Acceptance criteria

- [ ] repro no longer reproduces — every payload in the escape corpus raises `SandboxExecutionError`, none returns a subclass list or executes a syscall (record the corpus + results in Notes)
- [ ] `safe_exec` no longer calls `exec` on LLM output (tier 1) — or runs it under enforced rlimits + timeout + no-net (tier 2); state which tier shipped and why
- [ ] attribute access to any `__\w+__` name is rejected
- [ ] benign arithmetic/logic rules from `examples/` still evaluate correctly
- [ ] regression test `test_sandbox_escape_blocked_fixes_0001` added and passing, executing the real evaluator
- [ ] no new `eval`/`exec`/`pickle` introduced (cross-check tasks 0002, 0003)

## Notes

_record escape corpus, chosen tier, and the benign-rule sanity run here_

## Completion (2026-08-24)
Tier 1 shipped: `SandboxedInterpreter` now uses a deny-by-default hand-walked AST evaluator — no `exec`/`eval`/`compile` on the rule string. Allowlist: Assign(single Name), Constant, Name, BinOp(+ - * / // % **), UnaryOp, BoolOp, Compare, IfExp, and calls only to abs/min/max/sum/round/int/float/bool/len. Any `__\w+__` Name and every non-allowlisted node → `SandboxExecutionError`. Escape corpus all blocked; benign `result = max(a,b)*1.2` and `result = budget > 100000` still evaluate. Tests: `tests/test_sandbox_fixes_0001.py` (10). Full suite 105 passed.
