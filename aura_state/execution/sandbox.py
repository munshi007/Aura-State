import ast
import logging
import operator
import re
from typing import Dict, Any, Optional
from pydantic import BaseModel

try:  # openai is an optional heavy import; the sandbox no longer requires it.
    from openai import OpenAI
except Exception:  # pragma: no cover - openai always present in this repo
    OpenAI = Any  # type: ignore

from ..core.exceptions import AuraStateError

logger = logging.getLogger("aura_state.sandbox")

# Matches Python dunder identifiers (e.g. __class__, __subclasses__). Any Name
# matching this is rejected outright — it is the entry point for every known
# in-process exec escape gadget.
_DUNDER = re.compile(r"^__\w+__$")


class SandboxExecutionError(AuraStateError):
    """Raised when the sandbox rejects or fails to evaluate the generated rule."""
    pass


class CodeGeneration(BaseModel):
    python_code: str
    explanation: str


class SandboxedInterpreter:
    """
    Evaluates deterministic math/logic rules with a deny-by-default AST
    interpreter. There is NO ``exec``/``eval``/``compile`` of the rule text:
    the AST is walked by hand and only an explicit allowlist of node types and
    named calls is permitted. Everything else — attribute access, subscripting,
    lambdas, comprehensions, dunder names, imports — is rejected before any
    value is produced, which closes the classic
    ``().__class__.__bases__[0].__subclasses__()`` escape.
    """

    # AST call target name -> concrete safe implementation. This is the ONLY
    # set of callables reachable from a rule.
    _ALLOWED_CALLS = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "round": round,
        "int": int,
        "float": float,
        "bool": bool,
        "len": len,
    }

    _BINOPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    _UNARYOPS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
        ast.Not: operator.not_,
    }

    _COMPARE_OPS = {
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
    }

    def __init__(self, llm_client: Optional["OpenAI"] = None):
        # Retained for API/wiring compatibility with AuraEngine. The evaluator
        # itself never calls the model — rules are executed deterministically.
        self.client = llm_client

    # ── Node rejection helper ──────────────────────────────────────────────
    @staticmethod
    def _reject(node: ast.AST, reason: Optional[str] = None) -> "SandboxExecutionError":
        kind = type(node).__name__
        msg = f"Disallowed AST node in sandbox: {kind}"
        if reason:
            msg += f" ({reason})"
        return SandboxExecutionError(msg)

    # ── Expression evaluation (deny-by-default) ────────────────────────────
    def _eval(self, node: ast.AST, env: Dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, bool, str)):
                return node.value
            raise self._reject(node, f"unsupported constant type {type(node.value).__name__}")

        if isinstance(node, ast.Name):
            name = node.id
            if _DUNDER.match(name):
                raise self._reject(node, f"dunder name '{name}' forbidden")
            if name in env:
                return env[name]
            raise SandboxExecutionError(f"Unknown variable in sandbox rule: '{name}'")

        if isinstance(node, ast.BinOp):
            op = self._BINOPS.get(type(node.op))
            if op is None:
                raise self._reject(node.op, "unsupported binary operator")
            return op(self._eval(node.left, env), self._eval(node.right, env))

        if isinstance(node, ast.UnaryOp):
            op = self._UNARYOPS.get(type(node.op))
            if op is None:
                raise self._reject(node.op, "unsupported unary operator")
            return op(self._eval(node.operand, env))

        if isinstance(node, ast.BoolOp):
            values = [self._eval(v, env) for v in node.values]
            if isinstance(node.op, ast.And):
                result = True
                for v in values:
                    result = v
                    if not v:
                        break
                return result
            if isinstance(node.op, ast.Or):
                result = False
                for v in values:
                    result = v
                    if v:
                        break
                return result
            raise self._reject(node.op, "unsupported boolean operator")

        if isinstance(node, ast.Compare):
            left = self._eval(node.left, env)
            for op_node, comparator in zip(node.ops, node.comparators):
                op = self._COMPARE_OPS.get(type(op_node))
                if op is None:
                    raise self._reject(op_node, "unsupported comparison operator")
                right = self._eval(comparator, env)
                if not op(left, right):
                    return False
                left = right
            return True

        if isinstance(node, ast.IfExp):
            if self._eval(node.test, env):
                return self._eval(node.body, env)
            return self._eval(node.orelse, env)

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise self._reject(node, "only direct calls to allowlisted builtins are permitted")
            fname = node.func.id
            if fname not in self._ALLOWED_CALLS:
                raise SandboxExecutionError(f"Forbidden function call in sandbox: '{fname}'")
            if node.keywords:
                raise self._reject(node, "keyword arguments are not permitted")
            args = [self._eval(a, env) for a in node.args]
            return self._ALLOWED_CALLS[fname](*args)

        # Deny-by-default: Attribute, Subscript, Lambda, ListComp/SetComp/
        # DictComp/GeneratorExp, Starred, JoinedStr, and every other node type
        # land here and are rejected.
        raise self._reject(node)

    # ── Statement evaluation ───────────────────────────────────────────────
    def _exec_module(self, tree: ast.Module, env: Dict[str, Any]) -> None:
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign):
                if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                    raise self._reject(stmt, "only single simple-name assignment targets are allowed")
                target = stmt.targets[0].id
                if _DUNDER.match(target):
                    raise self._reject(stmt, f"dunder assignment target '{target}' forbidden")
                env[target] = self._eval(stmt.value, env)
            elif isinstance(stmt, ast.Expr):
                # Bare expression statement: evaluate (still allowlist-checked),
                # discard the value.
                self._eval(stmt.value, env)
            else:
                raise self._reject(stmt, "only assignments and expressions are allowed")

    def safe_exec(self, code_str: str, local_vars: Dict[str, Any]) -> Any:
        """
        Evaluate ``code_str`` against ``local_vars`` with the no-exec allowlist
        interpreter and return the value bound to ``result``.
        """
        try:
            tree = ast.parse(code_str, mode="exec")
        except SyntaxError as e:
            raise SandboxExecutionError(f"Invalid Python syntax: {e}")

        if not isinstance(tree, ast.Module):
            raise self._reject(tree)

        # Fresh environment seeded with the provided inputs. Assignments (e.g.
        # ``result = ...``) bind into this env; nothing leaks back to callers'
        # dicts and no builtins/globals are reachable.
        env: Dict[str, Any] = dict(local_vars)

        self._exec_module(tree, env)

        if "result" not in env:
            raise SandboxExecutionError(
                "The generated code failed to assign a value to the 'result' variable."
            )
        return env["result"]

    def compile_and_run(self, english_prompt: str, input_variables: Dict[str, Any]) -> Any:
        """
        Evaluate a sandbox rule (Python source such as ``"result = budget > 100000"``)
        against ``input_variables`` and return the value bound to ``result``.

        Rules are executed deterministically by :meth:`safe_exec`; there is no
        LLM code-generation or exec step. The parameter is named ``english_prompt``
        for backward compatibility with the previous public signature.
        """
        logger.info("Evaluating sandbox rule via no-exec allowlist interpreter.")
        return self.safe_exec(english_prompt, dict(input_variables))
