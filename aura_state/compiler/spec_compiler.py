"""
Design→spec compiler: emit a runtime contract from the verified graph.

The verification a workflow was designed and proven against -- the node
obligations (Z3), the graph's CTL verdicts, the per-node confidence -- is
compiled into a single portable, versioned `AuraContract`. Because the contract
is derived from the same typed design the engine runs, the specification is
faithful *by construction*: spec and implementation are one artifact and cannot
drift.

The native `AuraContract` JSON is the source of truth. A downstream enforcer
(e.g. a runtime monitor) can re-check the recorded obligations against live data;
`check_faithfulness` here asserts the contract and the in-loop verifier agree on
the same inputs. An LTLf/policy projection for a specific runtime is intended to
live in a *separate adapter*, not in this core module.
"""
import hashlib
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..verification.temporal_verifier import (
    verify_engine, _find_init_node, _find_sinks, PropertyResult,
)
from ..verification.proof_engine import prove_extraction

CONTRACT_SCHEMA_VERSION = 1


def _aura_version() -> str:
    try:
        from importlib.metadata import version
        return version("aura-state")
    except Exception:
        return "unknown"


class ContractError(Exception):
    """Raised when a contract artifact is malformed or the wrong schema."""


class NodeContract(BaseModel):
    name: str
    extracts: Optional[str] = None          # extraction schema class name, if any
    obligations: List[str] = Field(default_factory=list)   # Z3 obligations
    confidence: float = 0.9                 # nominal conformal coverage
    sandbox_rule: Optional[str] = None
    terminal: bool = False


class PropertyVerdict(BaseModel):
    description: str
    formula: str                            # CTL formula, as its repr
    verdict: str                            # "PROVEN" | "VIOLATED"


class AuraContract(BaseModel):
    """A versioned, content-addressable behavioral contract for a graph."""
    schema_version: int = CONTRACT_SCHEMA_VERSION
    entry_node: Optional[str] = None
    terminals: List[str] = Field(default_factory=list)
    nodes: List[NodeContract] = Field(default_factory=list)
    transitions: Dict[str, List[str]] = Field(default_factory=dict)
    properties: List[PropertyVerdict] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)   # provenance (hash, version, time)

    def _canonical_payload(self) -> dict:
        """Everything the contract *asserts*, excluding provenance metadata.

        This is what the content hash and structural equality are taken over,
        so re-emitting the same design (even at a different time) yields the
        same hash.
        """
        d = self.model_dump()
        d.pop("meta", None)
        return d

    def content_hash(self) -> str:
        blob = json.dumps(self._canonical_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "AuraContract":
        """Load + validate a contract. Fails CLOSED: malformed JSON, wrong
        schema version, or shape mismatch is rejected, never partially loaded."""
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ContractError(f"contract is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise ContractError("contract must be a JSON object")
        if data.get("schema_version") != CONTRACT_SCHEMA_VERSION:
            raise ContractError(
                f"unsupported contract schema_version {data.get('schema_version')!r} "
                f"(expected {CONTRACT_SCHEMA_VERSION})"
            )
        try:
            return cls.model_validate(data)
        except Exception as e:
            raise ContractError(f"contract failed validation: {e}") from e


def compile_contract(
    engine,
    *,
    properties: Optional[List[dict]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> AuraContract:
    """Compile a registered ``AuraEngine`` into an ``AuraContract``.

    Pulls structure from the engine's transitions/nodes, obligations and
    confidence from each Node, and -- if ``properties`` are given -- runs the
    CTL verifier and records each verdict as evidence. ``meta`` may carry
    provenance the compiler can't derive (e.g. an ``emitted_at`` timestamp);
    the content hash and ``aura_version`` are always stamped in.
    """
    nodes_dict = engine._nodes
    transitions = {k: list(v) for k, v in engine._transitions.items()}
    sinks = _find_sinks(nodes_dict, transitions)
    entry = _find_init_node(nodes_dict, transitions) if nodes_dict else None

    node_contracts: List[NodeContract] = []
    for name, node in nodes_dict.items():
        node_contracts.append(NodeContract(
            name=name,
            extracts=node.extracts.__name__ if node.extracts else None,
            obligations=list(node.obligations or []),
            confidence=node.confidence,
            sandbox_rule=node.sandbox_rule,
            terminal=name in sinks,
        ))

    prop_verdicts: List[PropertyVerdict] = []
    if properties:
        for vr in verify_engine(engine, properties, terminals=sinks):
            prop_verdicts.append(PropertyVerdict(
                description=vr.property_text,
                formula=vr.formula_repr,
                verdict="PROVEN" if vr.result == PropertyResult.PROVEN else "VIOLATED",
            ))

    m: Dict[str, Any] = dict(meta or {})
    m["aura_version"] = _aura_version()

    contract = AuraContract(
        entry_node=entry,
        terminals=sorted(sinks),
        nodes=node_contracts,
        transitions=transitions,
        properties=prop_verdicts,
        meta=m,
    )
    contract.meta["content_hash"] = contract.content_hash()
    return contract


def check_faithfulness(contract: AuraContract, node_name: str, data: Dict[str, Any]) -> bool:
    """Re-check a node's obligations straight from the contract.

    This is the faithfulness invariant in miniature: for the same inputs, the
    contract's obligations must return the same verdict the in-loop verifier
    would. A good extraction passes here and in the loop; a bad one fails both.
    (Task 0016 deepens this into metamorphic/differential coverage.)
    """
    node = next((n for n in contract.nodes if n.name == node_name), None)
    if node is None:
        raise ContractError(f"no node '{node_name}' in contract")
    return prove_extraction(data, node.obligations).verified


def diff_contracts(a: AuraContract, b: AuraContract) -> Dict[str, Any]:
    """Structural / obligation / property delta between two contracts.

    Empty dict means the two designs assert the same contract (ignoring
    provenance meta). Powers a design-time regression gate: a non-empty diff
    on a protected branch means the behavioral contract changed.
    """
    delta: Dict[str, Any] = {}
    if a.entry_node != b.entry_node:
        delta["entry_node"] = (a.entry_node, b.entry_node)
    if a.transitions != b.transitions:
        delta["transitions"] = (a.transitions, b.transitions)
    if sorted(a.terminals) != sorted(b.terminals):
        delta["terminals"] = (sorted(a.terminals), sorted(b.terminals))

    na = {n.name: n for n in a.nodes}
    nb = {n.name: n for n in b.nodes}
    node_changes: Dict[str, Any] = {}
    for name in set(na) | set(nb):
        if name not in na:
            node_changes[name] = ("added", nb[name].model_dump())
        elif name not in nb:
            node_changes[name] = ("removed", na[name].model_dump())
        elif na[name].model_dump() != nb[name].model_dump():
            node_changes[name] = (na[name].model_dump(), nb[name].model_dump())
    if node_changes:
        delta["nodes"] = node_changes

    pa = [(p.description, p.verdict) for p in a.properties]
    pb = [(p.description, p.verdict) for p in b.properties]
    if pa != pb:
        delta["properties"] = (pa, pb)

    return delta
