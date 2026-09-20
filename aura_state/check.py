"""Static analysis for agent designs — the engine behind ``aura-state check``.

Runs the real verifiers (taint dataflow, CTL reachability/completion, Z3
obligation consistency, and a content policy scan) over a flow spec and returns
a structured report. Fails CLOSED: anything unproven is a finding.

A flow is the same JSON the studio saves/exports:
    {"name": ..., "entry": ..., "edges": [[a, b], ...],
     "nodes": [{"id", "kind"|"type", "capability", "obligations", "fields",
                "sandbox_rule", "tool_name", "side_effect"}, ...],
     "invariants": [...]}
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .core.engine import AuraEngine, Node, CompiledTransition
from .verification.temporal_verifier import (
    reachability, eventual_completion, PropertyResult,
)
from .verification.proof_engine import prove_obligations_satisfiable
from .verification.trifecta import analyze_trifecta


SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


@dataclass
class Finding:
    check: str          # taint | reachability | completion | obligation | policy | invariant
    severity: str       # critical | high | medium | low
    node: Optional[str]
    message: str
    key: str = ""       # stable discriminator for baseline diffing (e.g. "source->sink")


@dataclass
class CheckReport:
    agent: str
    nodes: int
    verified: bool
    findings: List[Finding] = field(default_factory=list)
    summary: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent, "nodes": self.nodes, "verified": self.verified,
            "summary": self.summary,
            "findings": [f.__dict__ for f in self.findings],
        }


_POLICY_RULES = [
    ("secret.openai",  re.compile(r"sk-[A-Za-z0-9]{20,}"),                    "critical", "OpenAI-style API key"),
    ("secret.google",  re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),               "critical", "Google API key"),
    ("secret.aws",     re.compile(r"AKIA[0-9A-Z]{16}"),                       "critical", "AWS access key id"),
    ("secret.generic", re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{8,}"), "high", "hardcoded credential"),
    ("pii.ssn",        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                  "high", "US SSN"),
    ("pii.email",      re.compile(r"\b[\w.+\-]+@[\w\-]+\.[\w.\-]+\b"),        "medium", "email address"),
]


def _cap(n: Dict[str, Any]) -> str:
    """Effective taint capability — tool side-effect overrides for tool nodes."""
    kind = n.get("kind") or n.get("type") or "extract"
    if kind == "tool":
        return "plain" if n.get("side_effect") == "read" else "sink"
    if kind == "sanitizer":
        return "sanitizer"
    return n.get("capability", "plain")


def _build_engine(nodes: List[Dict[str, Any]], edges: List[List[str]]) -> AuraEngine:
    engine = AuraEngine()

    def _handler(self, user_text, extracted_data=None, memory=None):
        return "END", {}

    classes = {}
    for n in nodes:
        attrs: Dict[str, Any] = {"system_prompt": n["id"], "handle": _handler}
        cap = _cap(n)
        if cap == "untrusted":
            attrs["untrusted_source"] = True
        elif cap == "sink":
            attrs["dangerous_sink"] = True
        elif cap == "sanitizer":
            attrs["sanitizer"] = True
        if n.get("obligations"):
            attrs["obligations"] = list(n["obligations"])
        cls = type(n["id"], (Node,), attrs)
        classes[n["id"]] = cls
        engine.register(cls)
    for a, b in edges:
        if a in classes and b in classes:
            engine.connect([CompiledTransition(from_node=classes[a], to_node=classes[b])])
    return engine


def check_flow(flow: Dict[str, Any]) -> CheckReport:
    nodes = flow.get("nodes", [])
    edges = [list(e) for e in flow.get("edges", [])]
    name = flow.get("name", "agent")
    findings: List[Finding] = []

    if not nodes:
        return CheckReport(agent=name, nodes=0, verified=False,
                           findings=[Finding("structure", "high", None, "flow has no nodes")])
    # Shape validation — fail closed with a clear message, not a traceback.
    if not all(isinstance(n, dict) and isinstance(n.get("id"), str) and n["id"] for n in nodes):
        return CheckReport(agent=name, nodes=len(nodes), verified=False,
                           findings=[Finding("structure", "high", None, "every node must be an object with a non-empty string 'id'")])
    if len({n["id"] for n in nodes}) != len(nodes):
        return CheckReport(agent=name, nodes=len(nodes), verified=False,
                           findings=[Finding("structure", "high", None, "duplicate node ids")])
    if not all(isinstance(e, (list, tuple)) and len(e) == 2 for e in edges):
        return CheckReport(agent=name, nodes=len(nodes), verified=False,
                           findings=[Finding("structure", "high", None, "each edge must be a [from, to] pair")])

    engine = _build_engine(nodes, edges)
    ids = [n["id"] for n in nodes]
    outgoing = {a for a, _ in edges}
    leaves = [i for i in ids if i not in outgoing]
    entry = flow.get("entry") or (ids[0] if ids else None)

    # 1. Taint dataflow — untrusted source -> dangerous sink without a sanitizer.
    taint = engine.analyze_field_taint()
    if not taint.verified:
        for v in taint.violations:
            findings.append(Finding(
                "taint", "critical", v.sink,
                f"untrusted data from '{v.source}' can reach sink '{v.sink}'"
                f"{f' via field {v.field}' if v.field and v.field != '*' else ''} with no sanitizer — injection path",
                key=f"{v.source}->{v.sink}:{v.field or ''}"))

    # 2. CTL reachability — every node reachable from the DECLARED entry.
    props = [{"description": f"{i} reachable", "formula": reachability(i)} for i in ids]
    for i, vr in zip(ids, engine.verify(props, init_node=entry)):
        if vr.result != PropertyResult.PROVEN:
            findings.append(Finding("reachability", "high", i, f"node '{i}' is not reachable from the entry — dead code"))

    # 3. Completion — every path reaches a terminal (no accidental dead-ends).
    #    The `eventual_completion` CTL property (checked at the init state) is the
    #    real test here; a separate structural find_dead_ends over the same leaf
    #    set can never add anything (its terminals ARE the leaves), so it's gone.
    if leaves:
        for vr in engine.verify([{"description": "completes", "formula": eventual_completion(*leaves)}], init_node=entry):
            if vr.result != PropertyResult.PROVEN:
                findings.append(Finding("completion", "high", None, "some path never reaches a terminal — the agent can get stuck"))

    # 4. Obligation consistency — each node's obligations are jointly satisfiable.
    for n in nodes:
        obls = list(n.get("obligations", []))
        if obls:
            res = prove_obligations_satisfiable(obls)
            if not res.satisfiable:
                findings.append(Finding("obligation", "high", n["id"],
                                        f"node '{n['id']}' obligations are contradictory: {res.reason or ''}".strip()))

    # 4b. Lethal trifecta — private data + untrusted content + external comms on
    #     one reachable execution (prompt injection can exfiltrate). The flagship
    #     property: the injection leg reuses provenance/taint, the private leg is
    #     the payload.
    tri = analyze_trifecta(nodes, edges, flow.get("entry"))
    for tf in tri.findings:
        findings.append(Finding(
            "trifecta", "critical", tf.exfil,
            f"lethal trifecta closed — {tf.detail}. "
            f"Path: {' → '.join(tf.path)}. "
            f"Break it with a sanitizer between '{tf.untrusted}' and '{tf.exfil}', "
            f"or remove one of the three capabilities.",
            key=f"{tf.untrusted}->{tf.exfil}"))
    for uid in tri.unclassified:
        findings.append(Finding(
            "trifecta", "low", uid,
            f"tool '{uid}' has an unknown trust class — "
            f"tag it `data_class: private|untrusted|public` (or `exfil: true`) so the trifecta check is sound"))

    # 4c. Fail-closed on ambiguity: if there are unclassifiable tools AND both an
    #     untrusted source and a private read are reachable, an unknown tool COULD
    #     be the exfil leg — we cannot prove the trifecta is absent. Block it.
    have_untrusted = any("untrusted" in v for v in tri.roles.values())
    have_private = any("private" in v for v in tri.roles.values())
    if tri.unclassified and have_untrusted and have_private and tri.verified:
        preview = ", ".join(tri.unclassified[:4]) + ("…" if len(tri.unclassified) > 4 else "")
        findings.append(Finding(
            "trifecta", "high", None,
            f"cannot prove trifecta-free: {len(tri.unclassified)} tool(s) of unknown capability "
            f"({preview}) while untrusted input AND private data are both reachable — one may be an "
            f"exfiltration channel. Tag them with data_class/exfil to resolve."))

    # 4d. An import that recognized no tools analyzed nothing — don't report green.
    if flow.get("source") in ("mcp", "code") and not [n for n in nodes if (n.get("kind") or n.get("type")) == "tool"]:
        findings.append(Finding(
            "structure", "medium", None,
            "no tools were recognized in the import — nothing was analyzed (check the source / tool declarations)"))

    # 5. Agent-level invariants consistency.
    inv = list(flow.get("invariants", []))
    if inv and not prove_obligations_satisfiable(inv).satisfiable:
        findings.append(Finding("invariant", "high", None, "agent invariants are contradictory"))

    # 6. Policy scan — secrets / PII in prompts, rules, obligations.
    for n in nodes:
        spots = {"system_prompt": n.get("system_prompt", ""), "sandbox_rule": n.get("sandbox_rule", "")}
        for i, o in enumerate(n.get("obligations", [])):
            spots[f"obligation[{i}]"] = o
        for where, text in spots.items():
            for _, rx, sev, desc in _POLICY_RULES:
                if text and rx.search(str(text)):
                    findings.append(Finding("policy", sev, n["id"], f"{desc} in {n['id']}.{where}"))
                    break

    findings.sort(key=lambda f: SEVERITY_ORDER.get(f.severity, 9))
    blocking = [f for f in findings if f.severity in ("critical", "high")]
    verified = len(blocking) == 0
    summary = {
        "taint": "violated" if any(f.check == "taint" for f in findings) else "proven",
        "trifecta": "closed" if not tri.verified else "broken",
        "reachability": "violated" if any(f.check == "reachability" for f in findings) else "proven",
        "obligations": "violated" if any(f.check == "obligation" for f in findings) else "proven",
        "policy": f"{sum(1 for f in findings if f.check == 'policy')} flagged",
    }
    return CheckReport(agent=name, nodes=len(nodes), verified=verified, findings=findings, summary=summary)
