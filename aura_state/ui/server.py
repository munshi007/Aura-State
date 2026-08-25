"""
Local Aura Studio — a FastAPI app that runs the REAL verifiers on your machine.

No cloud, no API key. You build an agent graph in the browser; this backend runs
the actual Z3 proofs, CTL model checking, static taint analysis, and the
design->contract compiler, and returns the verdicts + counterexamples.

Launched via `aura-state ui`.
"""
import logging
import os
from typing import Any, Dict, List, Optional

logging.getLogger("aura_state").setLevel(logging.ERROR)   # quiet verbose verify logs

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The Aura Studio UI needs extra packages. Install them with:\n"
        "    pip install 'aura-state[ui]'"
    ) from e

from ..core.engine import AuraEngine, Node, CompiledTransition
from ..verification.temporal_verifier import reachability, PropertyResult
from ..verification.proof_engine import prove_obligations_satisfiable

_STATIC = os.path.join(os.path.dirname(__file__), "static")


# ── Request schema ──

class NodeSpec(BaseModel):
    id: str
    capability: str = "plain"          # plain | untrusted | sink | sanitizer
    obligations: List[str] = []


class GraphSpec(BaseModel):
    nodes: List[NodeSpec]
    edges: List[List[str]] = []        # [[from, to], ...]
    entry: Optional[str] = None


def _engine_from_spec(spec: GraphSpec) -> AuraEngine:
    """Construct a real AuraEngine from the browser's graph, so every check
    below is the actual framework verifier — not a re-implementation."""
    engine = AuraEngine()

    def _handler(self, user_text, extracted_data=None, memory=None):
        return "END", {}

    classes = {}
    for n in spec.nodes:
        attrs: Dict[str, Any] = {"system_prompt": n.id, "handle": _handler}
        if n.capability == "untrusted":
            attrs["untrusted_source"] = True
        elif n.capability == "sink":
            attrs["dangerous_sink"] = True
        elif n.capability == "sanitizer":
            attrs["sanitizer"] = True
        if n.obligations:
            attrs["obligations"] = list(n.obligations)
        cls = type(n.id, (Node,), attrs)
        classes[n.id] = cls
        engine.register(cls)

    for a, b in spec.edges:
        if a in classes and b in classes:
            engine.connect([CompiledTransition(from_node=classes[a], to_node=classes[b])])
    return engine


def create_app() -> "FastAPI":
    app = FastAPI(title="Aura Studio", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index():
        with open(os.path.join(_STATIC, "index.html")) as f:
            return f.read()

    @app.post("/api/verify")
    def verify(spec: GraphSpec):
        engine = _engine_from_spec(spec)

        # 1. Injection-safe dataflow (real static taint).
        taint = engine.analyze_field_taint()
        taint_out = {
            "verdict": "PROVEN" if taint.verified else "VIOLATED",
            "violations": [
                {"field": v.field, "source": v.source, "sink": v.sink, "path": v.path}
                for v in taint.violations
            ],
        }

        # 2. CTL: is every node reachable from the entry? (real model checking)
        props = [{"description": f"{n.id} reachable", "formula": reachability(n.id)}
                 for n in spec.nodes]
        ctl_out = []
        if props:
            for vr in engine.verify(props):
                ctl_out.append({
                    "property": vr.property_text,
                    "verdict": "PROVEN" if vr.result == PropertyResult.PROVEN else "VIOLATED",
                })

        # 3. Z3: are each node's obligations self-consistent? (real SMT)
        obligations_out = []
        for n in spec.nodes:
            if n.obligations:
                res = prove_obligations_satisfiable(n.obligations)
                obligations_out.append({
                    "node": n.id,
                    "consistent": res.satisfiable,
                    "reason": res.reason,
                })

        # 4. The compiled contract (real design->spec compiler).
        contract = engine.compile_contract(properties=props)

        return JSONResponse({
            "taint": taint_out,
            "ctl": ctl_out,
            "obligations": obligations_out,
            "contract": contract.model_dump(),
        })

    if os.path.isdir(_STATIC):
        app.mount("/static", StaticFiles(directory=_STATIC), name="static")
    return app
