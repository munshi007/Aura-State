"""A bundled, labeled benchmark for design-time agent verification.

`aura-state bench` runs this. Two honest measurements:

1. **Accuracy on labeled synthetics** — safe/vulnerable *pairs* we construct, so
   the ground truth is certain. We report recall + precision. Because the analysis
   is a *sound over-approximation* (fail-closed), recall on the trifecta/taint
   class is 100% by construction — it never silently passes a real one — and the
   interesting number is precision (how often a flag is a true positive).
2. **Import coverage on real-shaped agents** — the point no prototype matches:
   we ingest the agent's own shape (MCP tool surfaces / framework code) unmodified
   and return a verdict, where AgentFlow needs its DSL and FIDES needs its planner.

The corpus is intentionally small and readable — a benchmark you can audit, not a
black box. Extend it in `benchmarks/`.
"""
from __future__ import annotations

from typing import Any, Dict, List


def _n(id: str, **kw: Any) -> Dict[str, Any]:
    return {"id": id, **kw}


# ── Labeled synthetic corpus: safe/vulnerable pairs, ground truth certain ──
# Each: (name, flow, vulnerable?, the property the label is about).
_UNTRUSTED = {"kind": "tool", "tool_name": "web.fetch", "side_effect": "read", "capability": "untrusted"}
_PRIVATE = {"kind": "tool", "tool_name": "db.query", "side_effect": "read", "data_class": "private"}
_EXFIL = {"kind": "tool", "tool_name": "email.send", "side_effect": "external", "capability": "sink"}
_SANI = {"kind": "sanitizer"}

SYNTHETIC: List[Dict[str, Any]] = [
    # trifecta: untrusted + private + exfil on one path, no sanitizer -> vulnerable
    {"name": "trifecta-open", "vulnerable": True, "property": "trifecta",
     "flow": {"name": "trifecta-open", "entry": "U", "edges": [["U", "P"], ["P", "X"]],
              "nodes": [_n("U", **_UNTRUSTED), _n("P", **_PRIVATE), _n("X", **_EXFIL)]}},
    # same, with a sanitizer before the sink -> safe (the fix)
    {"name": "trifecta-sanitized", "vulnerable": False, "property": "trifecta",
     "flow": {"name": "trifecta-sanitized", "entry": "U", "edges": [["U", "P"], ["P", "S"], ["S", "X"]],
              "nodes": [_n("U", **_UNTRUSTED), _n("P", **_PRIVATE), _n("S", **_SANI), _n("X", **_EXFIL)]}},
    # taint: untrusted -> dangerous sink, no sanitizer -> vulnerable
    {"name": "taint-open", "vulnerable": True, "property": "taint",
     "flow": {"name": "taint-open", "entry": "U", "edges": [["U", "X"]],
              "nodes": [_n("U", **_UNTRUSTED), _n("X", **_EXFIL)]}},
    # taint: sanitizer between -> safe
    {"name": "taint-sanitized", "vulnerable": False, "property": "taint",
     "flow": {"name": "taint-sanitized", "entry": "U", "edges": [["U", "S"], ["S", "X"]],
              "nodes": [_n("U", **_UNTRUSTED), _n("S", **_SANI), _n("X", **_EXFIL)]}},
    # obligation: an extract node with a self-contradictory obligation -> vulnerable
    {"name": "obligation-contradiction", "vulnerable": True, "property": "obligation",
     "flow": {"name": "obligation-contradiction", "entry": "E", "edges": [],
              "nodes": [_n("E", kind="extract", capability="plain",
                           fields=[{"name": "amount", "type": "int"}],
                           obligations=["amount >= 100", "amount <= 10"])]}},
    # obligation: satisfiable -> safe
    {"name": "obligation-ok", "vulnerable": False, "property": "obligation",
     "flow": {"name": "obligation-ok", "entry": "E", "edges": [],
              "nodes": [_n("E", kind="extract", capability="plain",
                           fields=[{"name": "amount", "type": "int"}],
                           obligations=["amount >= 0", "amount <= 500"])]}},
    # untrusted + private but NO external sink -> safe (nothing can leave)
    {"name": "no-exfil-channel", "vulnerable": False, "property": "trifecta",
     "flow": {"name": "no-exfil-channel", "entry": "U", "edges": [["U", "P"]],
              "nodes": [_n("U", **_UNTRUSTED), _n("P", **_PRIVATE)]}},
    # fully clean linear agent -> safe
    {"name": "clean-linear", "vulnerable": False, "property": "clean",
     "flow": {"name": "clean-linear", "entry": "A", "edges": [["A", "B"]],
              "nodes": [_n("A", kind="extract", capability="plain"),
                        _n("B", kind="tool", tool_name="cache.read", side_effect="read")]}},
]


def real_shaped_paths() -> List[str]:
    """Real-shaped agents (MCP tool surfaces / framework compositions) shipped in
    examples/audit/ — used for the import-coverage measure. Returns paths that
    exist in a source checkout; empty from a bare wheel (coverage is a repo metric)."""
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(here, "examples", "audit")
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.endswith(".json"))


def run_bench() -> Dict[str, Any]:
    """Run the corpus. Returns per-case results + aggregate metrics."""
    from .check import check_flow

    cases = []
    tp = fp = tn = fn = 0
    for c in SYNTHETIC:
        r = check_flow(c["flow"])
        detected = not r.verified                    # blocking finding => flagged vulnerable
        expected = c["vulnerable"]
        if expected and detected: tp += 1
        elif expected and not detected: fn += 1      # a SILENT MISS — must be 0 (sound)
        elif not expected and detected: fp += 1      # a false positive (precision cost)
        else: tn += 1
        cases.append({"name": c["name"], "property": c["property"],
                      "expected_vulnerable": expected, "detected": detected,
                      "correct": expected == detected, "findings": len(r.findings)})

    recall = tp / (tp + fn) if (tp + fn) else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0

    # import coverage on real-shaped agents
    from .cli import _load_flow
    real = []
    for p in real_shaped_paths():
        import os
        name = os.path.basename(p)
        try:
            flow = _load_flow(p)
            nodes = [x for x in flow.get("nodes", []) if x.get("id") != "Agent"]
            r = check_flow(flow)
            real.append({"name": name, "imported": True, "nodes": len(nodes),
                         "verdict": "proven" if r.verified else "findings",
                         "findings": len(r.findings)})
        except Exception as e:
            real.append({"name": name, "imported": False, "error": str(e)[:80]})

    return {
        "synthetic": cases,
        "metrics": {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
                    "recall": recall, "precision": precision,
                    "silent_misses": fn},
        "real": real,
        "import_coverage": (sum(1 for r in real if r.get("imported")) / len(real)) if real else None,
    }
