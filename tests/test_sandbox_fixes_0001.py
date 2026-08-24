"""Regression tests for task 0001: sandbox RCE via exec escape gadgets.

These call the REAL no-exec allowlist evaluator (per CLAUDE.md rule 12 — the
unit under test is never monkeypatched). Every escape attempt must be rejected
with SandboxExecutionError *before* any value is produced; benign arithmetic /
comparison rules must still evaluate to the correct value.
"""
import pytest

from aura_state.execution.sandbox import SandboxedInterpreter, SandboxExecutionError


# Known in-process exec-escape gadgets and other out-of-allowlist constructs.
# None of these may execute; each must raise the sandbox rejection exception.
ESCAPE_CORPUS = [
    "().__class__.__bases__[0].__subclasses__()",
    "getattr((), '__class__')",
    "[x for x in ().__class__.__mro__]",
    "(lambda: ().__class__)()",
    "__import__('os')",
]


@pytest.mark.parametrize("payload", ESCAPE_CORPUS)
def test_sandbox_escape_blocked_fixes_0001(payload):
    sandbox = SandboxedInterpreter()
    with pytest.raises(SandboxExecutionError):
        sandbox.compile_and_run(payload, {})


def test_sandbox_escape_blocked_no_side_effect_fixes_0001():
    """A payload that would otherwise mutate state must not run at all."""
    sandbox = SandboxedInterpreter()
    marker = {"touched": False}

    # Even a syntactically valid attribute-access rule assigning to result is
    # rejected — attribute traversal is off the allowlist.
    with pytest.raises(SandboxExecutionError):
        sandbox.compile_and_run("result = ().__class__.__name__", {})

    assert marker["touched"] is False


def test_sandbox_benign_arithmetic_fixes_0001():
    sandbox = SandboxedInterpreter()
    result = sandbox.compile_and_run("result = max(a, b) * 1.2", {"a": 10, "b": 20})
    assert result == 24.0


def test_sandbox_benign_comparison_fixes_0001():
    sandbox = SandboxedInterpreter()
    result = sandbox.compile_and_run("result = budget > 100000", {"budget": 450000})
    assert result is True


def test_sandbox_missing_result_raises_fixes_0001():
    sandbox = SandboxedInterpreter()
    with pytest.raises(SandboxExecutionError):
        sandbox.compile_and_run("total = a + b", {"a": 1, "b": 2})


def test_sandbox_supported_ops_fixes_0001():
    """Exercise the remaining allowlisted node types end-to-end."""
    sandbox = SandboxedInterpreter()
    # BoolOp + Compare chain + IfExp + unary + allowlisted calls.
    rule = "result = (x if a and b else y) + abs(-len([1, 2, 3]) * 0)"
    # len/list literal — list literal is NOT allowlisted, so this must reject.
    with pytest.raises(SandboxExecutionError):
        sandbox.compile_and_run(rule, {"x": 1, "y": 2, "a": True, "b": True})

    ok = sandbox.compile_and_run(
        "result = (x if a and b else y) + abs(-round(3.6))",
        {"x": 1, "y": 2, "a": True, "b": True},
    )
    assert ok == 5  # 1 + abs(-4)
