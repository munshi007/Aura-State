"""Regression: JSON-Schema enum must be ENFORCED (was only stashed, not checked)."""
import pytest
from pydantic import ValidationError
from aura_state.compiler.schema_compiler import compile_pydantic_model


def test_enum_is_enforced_via_literal():
    Model = compile_pydantic_model("Ticket", {
        "properties": {"priority": {"type": "string", "enum": ["low", "high"]}},
        "required": ["priority"],
    })
    assert Model(priority="high").priority == "high"
    with pytest.raises(ValidationError):
        Model(priority="urgent")          # out-of-enum must fail
