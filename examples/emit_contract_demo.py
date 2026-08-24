#!/usr/bin/env python3
"""
Emit the runtime contract compiled from a verified graph — no API key needed.

The contract is *derived from the design* (node obligations + CTL verdicts +
confidence + structure), so it is faithful by construction. This is the seam a
downstream runtime monitor would enforce.

Run:
    python examples/emit_contract_demo.py
"""
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.getLogger("aura_state").setLevel(logging.ERROR)

from pydantic import BaseModel

from aura_state import (
    AuraEngine, Node, CompiledTransition,
    reachability, eventual_completion,
    check_faithfulness,
)


class Lead(BaseModel):
    budget: int = 450000
    bedrooms: int = 3
    price_per_bed: int = 150000
    total: int = 450000


class QualifyLead(Node):
    system_prompt = "Extract and qualify a real-estate lead."
    extracts = Lead
    obligations = ["budget > 0", "total == bedrooms * price_per_bed"]
    confidence = 0.95

    def handle(self, user_text, extracted_data=None, memory=None):
        return "RouteLead", extracted_data.model_dump()


class RouteLead(Node):
    system_prompt = "Route the qualified lead."

    def handle(self, user_text, extracted_data=None, memory=None):
        return "END", {}


def main():
    engine = AuraEngine()
    engine.register(QualifyLead, RouteLead)
    engine.connect([CompiledTransition(from_node=QualifyLead, to_node=RouteLead)])

    properties = [
        {"description": "RouteLead is reachable", "formula": reachability("RouteLead")},
        {"description": "the flow completes", "formula": eventual_completion("RouteLead")},
    ]

    contract = engine.compile_contract(properties=properties, meta={"emitted_at": "2026-08-24T00:00:00Z"})

    print("=" * 68)
    print("  AuraContract — compiled from the verified graph")
    print("=" * 68)
    print(contract.to_json())

    print("\n" + "-" * 68)
    print("  Faithfulness: the contract's obligations agree with the loop")
    print("-" * 68)
    good = {"budget": 450000, "bedrooms": 3, "price_per_bed": 150000, "total": 450000}
    bad = {"budget": 450000, "bedrooms": 3, "price_per_bed": 150000, "total": 999}
    print(f"  good lead {good['total']=} -> accepted : {check_faithfulness(contract, 'QualifyLead', good)}")
    print(f"  bad  lead {bad['total']=} -> accepted : {check_faithfulness(contract, 'QualifyLead', bad)}")
    print(f"\n  content hash: {contract.meta['content_hash'][:16]}…  (content-addressable; re-emitting the same design reproduces it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
