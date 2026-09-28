"""A compliance-grade, independently verifiable proof certificate.

`make_certificate` records the agent design + the verdict `check_flow` produced;
`verify_certificate` re-checks it **without trusting the operator**: it recomputes
the design hash (tamper on the design), the verdict hash (tamper on the claimed
result), and — the real guarantee — **re-runs the verifier on the recorded design**
and confirms the recorded verdict matches. So a certificate that claims "verified"
can be independently confirmed by anyone with aura-state, and a doctored one is
caught.

This is the design-time / pre-deployment counterpart to runtime proof-carrying /
compliance-receipt work (Proof-Carrying Agent Actions, arXiv:2606.04104; the
Microsoft Agent Governance Toolkit's verifiable receipts). It maps to the EU AI
Act Art. 12 need for *evidence, not documentation*, and to ISO/IEC 42001 as a
technical evidence artifact.

Not a cryptographic signature (no key): tamper-evidence comes from the content
hashes plus independent re-verification, which is what makes it trustless.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def _hash(obj: Any) -> str:
    return hashlib.sha256(_canon(obj).encode()).hexdigest()


def _design_of(flow: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": flow.get("name", "agent"),
        "entry": flow.get("entry"),
        "edges": [list(e) for e in flow.get("edges", [])],
        "nodes": flow.get("nodes", []),
        "invariants": list(flow.get("invariants", [])),
        "manifest": flow.get("manifest"),
    }


def make_certificate(flow: Dict[str, Any], *, version: str, timestamp: str) -> Dict[str, Any]:
    """Emit a verifiable certificate for a flow. `timestamp` is passed in (scripts
    can't read the clock) — an ISO-8601 string."""
    from .check import check_flow

    design = _design_of(flow)
    report = check_flow(flow)
    verdict = {
        "verified": report.verified,
        "summary": report.summary,
        "findings": [f.__dict__ for f in report.findings],
    }
    return {
        "aura_certificate": "1.0",
        "generated": timestamp,
        "aura_state_version": version,
        "design": design,
        "design_hash": _hash(design),
        "verdict": verdict,
        "verdict_hash": _hash(verdict),
        "standards": {
            "eu_ai_act_article_12": "design-time verification evidence; independently re-verifiable",
            "iso_iec_42001": "AI management system technical evidence artifact",
        },
    }


def verify_certificate(cert: Dict[str, Any]) -> Dict[str, Any]:
    """Independently re-check a certificate. Returns {valid, problems, recomputed}."""
    from .check import check_flow

    problems: List[str] = []
    if not isinstance(cert, dict) or cert.get("aura_certificate") != "1.0":
        return {"valid": False, "problems": ["not an aura_certificate 1.0 document"], "recomputed": None}

    design = cert.get("design")
    if not isinstance(design, dict):
        return {"valid": False, "problems": ["certificate has no design to re-verify"], "recomputed": None}

    # 1. tamper-evidence: recorded hashes must match the recorded content.
    if _hash(design) != cert.get("design_hash"):
        problems.append("design hash mismatch — the recorded design was tampered with")
    if _hash(cert.get("verdict")) != cert.get("verdict_hash"):
        problems.append("verdict hash mismatch — the recorded verdict was tampered with")

    # 2. the real guarantee: re-run the verifier on the recorded design and confirm
    #    the recorded verdict. Catches a certificate claiming "verified" for a design
    #    that does not actually verify — without trusting whoever issued it.
    rep = check_flow(design)
    recomputed_findings = [f.__dict__ for f in rep.findings]
    recomputed = {"verified": rep.verified, "summary": rep.summary, "findings": recomputed_findings}
    claimed = cert.get("verdict") or {}
    if rep.verified != claimed.get("verified"):
        problems.append(f"re-verification disagrees: recomputed verified={rep.verified}, "
                        f"certificate claims {claimed.get('verified')}")
    if rep.summary != claimed.get("summary"):
        problems.append("re-verification disagrees: recomputed summary differs from the certificate")
    # The findings ARE the evidence — recompute and compare them too, so a cert whose
    # verdict hashes match but whose evidence was rewritten (hidden/altered finding)
    # is still caught. check_flow is deterministic, so an honest cert matches exactly.
    if recomputed_findings != (claimed.get("findings") or []):
        problems.append("re-verification disagrees: recorded findings do not match the recomputed evidence")

    return {"valid": not problems, "problems": problems, "recomputed": recomputed}
