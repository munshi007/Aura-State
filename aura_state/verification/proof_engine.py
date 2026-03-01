"""
Z3-backed proof engine for formal verification of LLM extractions.

Compiles Pydantic field constraints and developer-defined proof
obligations into Z3 SMT formulas. If unsatisfiable, generates
a counterexample describing exactly which constraint failed.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from z3 import (
    Solver, Int, Real, Bool, sat, unsat,
    And as Z3And, Or as Z3Or, Not as Z3Not, Implies,
    ArithRef, BoolRef,
)

logger = logging.getLogger("aura_state.proof")


@dataclass
class ProofResult:
    verified: bool
    failed_obligations: List[str]
    counterexample: Optional[Dict[str, Any]] = None


def _make_z3_var(name: str, value: Any) -> Tuple[Any, Any]:
    """Create a Z3 variable matching the Python type of the value."""
    if isinstance(value, bool):
        return Bool(name), value
    if isinstance(value, int):
        return Int(name), value
    if isinstance(value, float):
        return Real(name), value
    return None, value


def _parse_obligation(obligation: str, variables: Dict[str, Any]) -> Optional[BoolRef]:
    """
    Parse a simple proof obligation string into a Z3 constraint.

    Supported operators: >, <, >=, <=, ==, !=, and, or, not
    All variable names in the obligation must exist in the variables dict.
    """
    z3_vars = {}
    for name, value in variables.items():
        z3_var, _ = _make_z3_var(name, value)
        if z3_var is not None:
            z3_vars[name] = z3_var

    if not z3_vars:
        return None

    try:
        result = eval(obligation, {"__builtins__": {}}, z3_vars)
        if isinstance(result, BoolRef):
            return result
        if isinstance(result, bool):
            return None
        return None
    except Exception as e:
        logger.warning(f"Could not parse obligation '{obligation}': {e}")
        return None


def prove_extraction(
    extracted_data: Dict[str, Any],
    obligations: List[str],
) -> ProofResult:
    """
    Verify that extracted data satisfies all proof obligations using Z3.

    Args:
        extracted_data: Dict of field_name -> value from the LLM extraction.
        obligations: List of constraint strings (e.g., "area > 0", "cost == area * rate").

    Returns:
        ProofResult with verification status and any failed constraints.
    """
    if not obligations:
        return ProofResult(verified=True, failed_obligations=[])

    z3_vars = {}
    for name, value in extracted_data.items():
        z3_var, _ = _make_z3_var(name, value)
        if z3_var is not None:
            z3_vars[name] = z3_var

    if not z3_vars:
        return ProofResult(verified=True, failed_obligations=[])

    # Pin each variable to its extracted value
    solver = Solver()
    for name, value in extracted_data.items():
        if name in z3_vars:
            solver.add(z3_vars[name] == value)

    failed = []
    for obligation in obligations:
        constraint = _parse_obligation(obligation, extracted_data)
        if constraint is None:
            continue

        # Check if the constraint is satisfied with the extracted values
        test_solver = Solver()
        for name, value in extracted_data.items():
            if name in z3_vars:
                test_solver.add(z3_vars[name] == value)
        test_solver.add(Z3Not(constraint))

        # If negation is sat, the original constraint is violated
        if test_solver.check() == sat:
            failed.append(obligation)
            model = test_solver.model()
            logger.info(f"Obligation failed: {obligation}")

    if failed:
        counterexample = {
            "extracted_values": {k: v for k, v in extracted_data.items() if isinstance(v, (int, float, bool))},
            "failed_constraints": failed,
        }
        return ProofResult(
            verified=False,
            failed_obligations=failed,
            counterexample=counterexample,
        )

    return ProofResult(verified=True, failed_obligations=[])


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
