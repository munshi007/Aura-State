"""Task 0015: the design→spec compiler emits a faithful, portable contract."""
from pydantic import BaseModel

from aura_state import (
    AuraEngine, Node, CompiledTransition,
    reachability, eventual_completion,
    compile_contract, diff_contracts, check_faithfulness,
    AuraContract, ContractError,
)


class Quote(BaseModel):
    area: int = 100
    rate: int = 3
    total: int = 300


class Price(Node):
    system_prompt = "price it"
    extracts = Quote
    obligations = ["total == area * rate", "area > 0"]
    confidence = 0.95

    def handle(self, user_text, extracted_data=None, memory=None):
        return "Done", {}


class Done(Node):
    system_prompt = "done"

    def handle(self, user_text, extracted_data=None, memory=None):
        return "END", {}


def _engine():
    e = AuraEngine()
    e.register(Price, Done)
    e.connect([CompiledTransition(from_node=Price, to_node=Done)])
    return e


PROPS = [
    {"description": "Done is reachable", "formula": reachability("Done")},
    {"description": "Price completes", "formula": eventual_completion("Done")},
]


def test_contract_captures_structure_and_obligations_fixes_0015():
    c = _engine().compile_contract(properties=PROPS)
    price = next(n for n in c.nodes if n.name == "Price")
    assert price.obligations == ["total == area * rate", "area > 0"]
    assert price.confidence == 0.95
    assert price.extracts == "Quote"
    assert c.transitions["Price"] == ["Done"]
    assert c.entry_node == "Price"
    assert "Done" in c.terminals
    # CTL verdicts recorded as evidence.
    assert len(c.properties) == 2
    assert all(p.verdict in ("PROVEN", "VIOLATED") for p in c.properties)
    # Content-addressable.
    assert len(c.meta["content_hash"]) == 64


def test_contract_round_trips_fixes_0015():
    c = _engine().compile_contract(properties=PROPS)
    restored = AuraContract.from_json(c.to_json())
    assert restored == c
    assert restored.content_hash() == c.content_hash()


def test_from_json_fails_closed_fixes_0015():
    import pytest
    with pytest.raises(ContractError):
        AuraContract.from_json("not json {{{")
    with pytest.raises(ContractError):
        AuraContract.from_json('{"schema_version": 999, "nodes": []}')


def test_faithfulness_contract_agrees_with_loop_fixes_0015():
    # The contract's obligations must reproduce the in-loop verdict on the same
    # inputs: a good extraction passes, a bad one fails -- straight from the
    # contract, no engine.
    c = _engine().compile_contract()
    assert check_faithfulness(c, "Price", {"area": 100, "rate": 3, "total": 300}) is True
    assert check_faithfulness(c, "Price", {"area": 100, "rate": 3, "total": 999}) is False


def test_diff_detects_obligation_change_fixes_0015():
    a = _engine().compile_contract()

    class Price2(Node):
        system_prompt = "price it"
        extracts = Quote
        obligations = ["total == area * rate"]  # dropped "area > 0"

        def handle(self, user_text, extracted_data=None, memory=None):
            return "Done", {}

    e2 = AuraEngine()
    e2.register(Price2, Done)
    e2.connect([CompiledTransition(from_node=Price2, to_node=Done)])
    # Register under the same name so structure lines up; rename class node key.
    e2._nodes["Price"] = e2._nodes.pop("Price2")
    e2._transitions["Price"] = e2._transitions.pop("Price2")
    b = e2.compile_contract()

    assert diff_contracts(a, a) == {}          # identical -> empty
    d = diff_contracts(a, b)
    assert "nodes" in d and "Price" in d["nodes"]   # obligation change caught
