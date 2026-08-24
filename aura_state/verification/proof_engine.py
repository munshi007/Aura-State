"""
Z3-backed proof engine for formal verification of LLM extractions.

Compiles Pydantic field constraints and developer-defined proof
obligations into Z3 SMT formulas. If unsatisfiable, generates
a counterexample describing exactly which constraint failed.

Obligations are parsed to a Python AST and compiled to typed Z3 ops
through an explicit allowlist of node types (no ``eval``/``exec``). Any
node outside the allowlist is rejected, and any obligation that cannot
be compiled or bound to the extracted data is reported as *unproven* --
the engine fails CLOSED: an obligation is only ever treated as satisfied
when Z3 proves it holds for the extracted values.
"""
import ast
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from z3 import (
    Solver, Int, Real, Bool, sat,
    And as Z3And, Or as Z3Or, Not as Z3Not,
    BoolRef,
)

logger = logging.getLogger("aura_state.proof")


class ObligationError(Exception):
    """Raised when an obligation cannot be compiled to a Z3 constraint."""


@dataclass
class ProofResult:
    verified: bool
    failed_obligations: List[str]
    # Obligations that could not be compiled or bound to the data. These are
    # NOT proven and NOT violated -- they are uncovered, and their presence
    # forces verified=False (fail closed).
    unproven_obligations: List[str] = field(default_factory=list)
    counterexample: Optional[Dict[str, Any]] = None


def _make_z3_var(name: str, value: Any) -> Tuple[Any, Any]:
    """Create a Z3 variable matching the Python type of the value."""
    # bool must be checked before int -- bool is a subclass of int.
    if isinstance(value, bool):
        return Bool(name), value
    if isinstance(value, int):
        return Int(name), value
    if isinstance(value, float):
        return Real(name), value
    return None, value


# --- Obligation compiler: Python AST -> Z3, allowlist only -------------------

_CMP_OPS = (ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq)
_BIN_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv)


def _apply_cmp(op: ast.cmpop, left: Any, right: Any) -> BoolRef:
    if isinstance(op, ast.Gt):
        return left > right
    if isinstance(op, ast.GtE):
        return left >= right
    if isinstance(op, ast.Lt):
        return left < right
    if isinstance(op, ast.LtE):
        return left <= right
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.NotEq):
        return left != right
    raise ObligationError(f"unsupported comparison: {type(op).__name__}")


def _apply_bin(op: ast.operator, left: Any, right: Any) -> Any:
    if isinstance(op, ast.Add):
        return left + right
    if isinstance(op, ast.Sub):
        return left - right
    if isinstance(op, ast.Mult):
        return left * right
    if isinstance(op, ast.Div) or isinstance(op, ast.FloorDiv):
        return left / right
    if isinstance(op, ast.Mod):
        return left % right
    if isinstance(op, ast.Pow):
        return left ** right
    raise ObligationError(f"unsupported operator: {type(op).__name__}")


def _compile_node(node: ast.AST, z3_vars: Dict[str, Any]) -> Any:
    """Recursively compile an allowlisted AST node to a Z3 expression."""
    if isinstance(node, ast.Expression):
        return _compile_node(node.body, z3_vars)

    if isinstance(node, ast.BoolOp):
        parts = [_compile_node(v, z3_vars) for v in node.values]
        if isinstance(node.op, ast.And):
            return Z3And(*parts)
        if isinstance(node.op, ast.Or):
            return Z3Or(*parts)
        raise ObligationError(f"unsupported boolean op: {type(node.op).__name__}")

    if isinstance(node, ast.UnaryOp):
        operand = _compile_node(node.operand, z3_vars)
        if isinstance(node.op, ast.Not):
            return Z3Not(operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise ObligationError(f"unsupported unary op: {type(node.op).__name__}")

    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _BIN_OPS):
            raise ObligationError(f"unsupported operator: {type(node.op).__name__}")
        return _apply_bin(
            node.op,
            _compile_node(node.left, z3_vars),
            _compile_node(node.right, z3_vars),
        )

    if isinstance(node, ast.Compare):
        # Support chained comparisons (a < b <= c) by AND-ing each pair.
        clauses = []
        prev = _compile_node(node.left, z3_vars)
        for op, comparator in zip(node.ops, node.comparators):
            if not isinstance(op, _CMP_OPS):
                raise ObligationError(f"unsupported comparison: {type(op).__name__}")
            right = _compile_node(comparator, z3_vars)
            clauses.append(_apply_cmp(op, prev, right))
            prev = right
        return Z3And(*clauses) if len(clauses) > 1 else clauses[0]

    if isinstance(node, ast.Name):
        if node.id in z3_vars:
            return z3_vars[node.id]
        raise ObligationError(f"unknown or non-numeric variable '{node.id}'")

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or isinstance(node.value, (int, float)):
            return node.value
        raise ObligationError(f"unsupported constant: {node.value!r}")

    raise ObligationError(f"unsupported syntax: {type(node).__name__}")


def _compile_obligation(obligation: str, z3_vars: Dict[str, Any]) -> BoolRef:
    """
    Compile an obligation string to a Z3 BoolRef via an AST allowlist.

    Supported: >, <, >=, <=, ==, !=, and, or, not, + - * / % **, chained
    comparisons, numeric literals, and variable names present in z3_vars.

    Raises ObligationError on anything else, or if the top-level result is
    not a boolean constraint. Never returns a non-boolean or None.
    """
    try:
        tree = ast.parse(obligation, mode="eval")
    except SyntaxError as e:
        raise ObligationError(f"syntax error: {e}") from e

    result = _compile_node(tree, z3_vars)
    if not isinstance(result, BoolRef):
        raise ObligationError(
            "obligation did not evaluate to a boolean constraint "
            f"(got {type(result).__name__})"
        )
    return result


def prove_extraction(
    extracted_data: Dict[str, Any],
    obligations: List[str],
) -> ProofResult:
    """
    Verify that extracted data satisfies all proof obligations using Z3.

    Fails CLOSED: an obligation is only marked satisfied when Z3 proves it
    holds for the extracted values. An obligation that cannot be compiled or
    bound to the data is reported as unproven and forces verified=False.

    Args:
        extracted_data: Dict of field_name -> value from the LLM extraction.
        obligations: List of constraint strings (e.g., "area > 0", "cost == area * rate").

    Returns:
        ProofResult with verification status, violated constraints, and any
        constraints that could not be proven either way.
    """
    if not obligations:
        return ProofResult(verified=True, failed_obligations=[])

    z3_vars: Dict[str, Any] = {}
    for name, value in extracted_data.items():
        z3_var, _ = _make_z3_var(name, value)
        if z3_var is not None:
            z3_vars[name] = z3_var

    failed: List[str] = []
    unproven: List[str] = []

    for obligation in obligations:
        try:
            constraint = _compile_obligation(obligation, z3_vars)
        except ObligationError as e:
            # Cannot express the obligation over this data -> unproven, not passed.
            logger.warning(f"Obligation unproven '{obligation}': {e}")
            unproven.append(obligation)
            continue

        # Pin each numeric variable to its extracted value, then check whether
        # the constraint can be violated. If Not(constraint) is sat under the
        # pinning, the constraint does not hold for the extracted values.
        solver = Solver()
        for name in z3_vars:
            solver.add(z3_vars[name] == extracted_data[name])
        solver.add(Z3Not(constraint))

        if solver.check() == sat:
            failed.append(obligation)
            logger.info(f"Obligation failed: {obligation}")

    verified = not failed and not unproven
    counterexample = None
    if not verified:
        counterexample = {
            "extracted_values": {
                k: v for k, v in extracted_data.items()
                if isinstance(v, (int, float, bool))
            },
            "failed_constraints": failed,
            "unproven_constraints": unproven,
        }

    return ProofResult(
        verified=verified,
        failed_obligations=failed,
        unproven_obligations=unproven,
        counterexample=counterexample,
    )


def prove_consistency(
    extracted_data: Dict[str, Any],
    relationships: List[str],
) -> ProofResult:
    """
    Check whether cross-field relationships hold.

    Example relationships:
        ["total_cost == unit_cost * quantity", "margin >= 0"]
    """
    return prove_extraction(extracted_data, relationships)
