"""
Local Aura Studio — a FastAPI app that runs the REAL verifiers on your machine.

No cloud, no API key. You build an agent graph in the browser; this backend runs
the actual Z3 proofs, CTL model checking, static taint analysis, and the
design->contract compiler, and returns the verdicts + counterexamples.

Launched via `aura-state ui`.
"""
import json
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
from ..verification.proof_engine import prove_obligations_satisfiable, prove_extraction
from ..verification.conformal import conformal_interval
from ..verification.pipeline_conformal import PipelineConformal
from ..verification.risk_control import RiskController

_STATIC = os.path.join(os.path.dirname(__file__), "static")

# OpenAI-compatible providers for the Live Agent module. Keys are read from the
# server's environment (this server runs locally, on the user's machine).
_PROVIDERS = {
    "ollama":   {"env": None,               "base_url": "http://localhost:11434/v1",                                   "model": "qwen2.5:0.5b",   "mode": "JSON"},
    "openai":   {"env": "OPENAI_API_KEY",   "base_url": None,                                                          "model": "gpt-4o-mini",    "mode": "TOOLS"},
    "gemini":   {"env": "GOOGLE_API_KEY",   "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",    "model": "gemini-2.0-flash","mode": "JSON"},
    "deepseek": {"env": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com",                                    "model": "deepseek-chat",  "mode": "TOOLS"},
}


def _clean(x):
    """JSON-safe: replace inf/None threshold sentinels."""
    import math
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return None
    return x


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

    # ── Live Agent module ──
    @app.get("/api/providers")
    def providers():
        return [
            {"name": n, "model": c["model"], "available": (c["env"] is None or bool(os.environ.get(c["env"]))),
             "needs": c["env"]}
            for n, c in _PROVIDERS.items()
        ]

    class AgentReq(BaseModel):
        provider: str = "ollama"
        model: Optional[str] = None
        prompt: str
        fields: List[Dict[str, str]] = []      # [{name,type,description}]
        obligations: List[str] = []

    @app.post("/api/agent")
    def agent(req: AgentReq):
        cfg = _PROVIDERS.get(req.provider)
        if not cfg:
            return JSONResponse({"error": f"unknown provider '{req.provider}'"}, status_code=400)
        if cfg["env"] and not os.environ.get(cfg["env"]):
            return {"error": f"set {cfg['env']} in the environment to use {req.provider}"}
        import instructor
        from openai import OpenAI
        from pydantic import create_model

        typemap = {"str": str, "int": int, "float": float, "bool": bool}
        defs = {f["name"]: (typemap.get(f.get("type", "str"), str), ...) for f in req.fields if f.get("name")}
        Model = create_model("Extracted", **(defs or {"result": (str, ...)}))
        client = OpenAI(api_key=os.environ.get(cfg["env"], "ollama") if cfg["env"] else "ollama", base_url=cfg["base_url"])
        patched = instructor.from_openai(client, mode=getattr(instructor.Mode, cfg["mode"]))
        model = req.model or cfg["model"]
        try:
            obj = patched.chat.completions.create(
                model=model, response_model=Model, max_retries=1,
                messages=[{"role": "user", "content": req.prompt}],
            )
        except Exception as e:
            return {"error": f"{req.provider} call failed: {str(e)[:220]}"}
        data = obj.model_dump()
        proof = prove_extraction(data, req.obligations) if req.obligations else None
        return {
            "extracted": data, "provider": req.provider, "model": model,
            "verified": (proof.verified if proof else None),
            "failed": (proof.failed_obligations if proof else []),
            "counterexample": (proof.counterexample if proof else None),
        }

    # ── Uncertainty (conformal / PASC) module ──
    class ConformalReq(BaseModel):
        values: List[float] = []
        predictions: List[float] = []
        truths: List[float] = []
        confidence: float = 0.9

    @app.post("/api/conformal")
    def conformal(req: ConformalReq):
        if req.predictions and req.truths and len(req.predictions) == len(req.truths):
            pc = PipelineConformal(confidence=req.confidence).calibrate(req.predictions, req.truths)
            cov = sum(pc.covers(p, t) for p, t in zip(req.predictions, req.truths)) / len(req.truths)
            return {"mode": "pasc", "q_hat": _clean(pc.q_hat), "calibrated": pc.calibrated,
                    "coverage": cov, "min_samples": pc.min_samples()}
        ci = conformal_interval(req.values, confidence=req.confidence)
        return {"mode": "interval", "lower": _clean(ci.lower), "upper": _clean(ci.upper),
                "point": _clean(getattr(ci, "point_estimate", None)),
                "calibrated": getattr(ci, "calibrated", None),
                "confidence": _clean(getattr(ci, "confidence", None)),
                "n": len(req.values)}

    # ── Risk-controlled abstention module ──
    class RiskReq(BaseModel):
        scores: List[float]
        correct: List[bool]
        epsilon: float = 0.1
        test_score: Optional[float] = None

    @app.post("/api/risk")
    def risk(req: RiskReq):
        ctrl = RiskController(epsilon=req.epsilon).calibrate(req.scores, req.correct)
        acted = [(s, c) for s, c in zip(req.scores, req.correct) if ctrl.should_act(s)]
        far = (sum(1 for s, c in acted if not c) / len(req.scores)) if req.scores else 0.0
        out = {"calibrated": ctrl.calibrated, "can_act": ctrl.can_act,
               "threshold": _clean(ctrl.threshold), "epsilon": req.epsilon,
               "realized_false_action_rate": far, "acted": len(acted), "n": len(req.scores)}
        if req.test_score is not None:
            out["decision"] = "act" if ctrl.should_act(req.test_score) else "abstain"
            out["test_score"] = req.test_score
        return out

    # ── Monitor: your real agent streams verified outputs here (SDK -> studio) ──
    feed: List[Dict[str, Any]] = []

    class IngestReq(BaseModel):
        node: str = "extraction"
        source: Optional[str] = None
        data: Dict[str, Any] = {}
        obligations: List[str] = []

    @app.post("/api/ingest")
    def ingest(req: IngestReq):
        import datetime
        r = prove_extraction(req.data, req.obligations) if req.obligations else None
        ev = {"node": req.node, "source": req.source, "data": req.data,
              "obligations": req.obligations,
              "verified": (r.verified if r else None),
              "failed": (r.failed_obligations if r else []),
              "ts": datetime.datetime.now().strftime("%H:%M:%S")}
        feed.append(ev)
        del feed[:-300]
        return {"verified": ev["verified"], "failed": ev["failed"]}

    @app.get("/api/feed")
    def get_feed():
        return list(reversed(feed))[:120]

    @app.post("/api/feed/clear")
    def clear_feed():
        feed.clear()
        return {"ok": True}

    # ── Data import: bulk-verify a real dataset against obligations ──
    class DatasetReq(BaseModel):
        records: List[Dict[str, Any]]
        obligations: List[str]

    @app.post("/api/verify_dataset")
    def verify_dataset(req: DatasetReq):
        import time
        t0 = time.time()
        passed = 0
        violations: List[Dict[str, Any]] = []
        for i, rec in enumerate(req.records):
            r = prove_extraction(rec, req.obligations)
            if r.verified:
                passed += 1
            elif len(violations) < 25:
                violations.append({"row": i, "record": rec,
                                   "failed": r.failed_obligations, "unproven": r.unproven_obligations})
        dt = max(time.time() - t0, 1e-6)
        return {"total": len(req.records), "passed": passed,
                "failed": len(req.records) - passed, "violations": violations,
                "obligations": len(req.obligations),
                "rate": round(len(req.records) / dt)}

    # ── Prove module (Z3 point-check + spec consistency on any data) ──
    class ProveReq(BaseModel):
        data: Dict[str, Any] = {}
        obligations: List[str] = []

    @app.post("/api/prove")
    def prove(req: ProveReq):
        r = prove_extraction(req.data, req.obligations)
        sat = prove_obligations_satisfiable(req.obligations)
        return {
            "verified": r.verified,
            "failed": r.failed_obligations,
            "unproven": r.unproven_obligations,
            "counterexample": r.counterexample,
            "consistent": sat.satisfiable,
            "witness": sat.witness,
            "reason": sat.reason,
        }

    # ── Build + Run: construct a real engine from a flow and run it end to end ──
    def _client_for(provider: str):
        import instructor
        from openai import OpenAI
        cfg = _PROVIDERS.get(provider) or _PROVIDERS["ollama"]
        if cfg["env"] and not os.environ.get(cfg["env"]):
            raise RuntimeError(f"set {cfg['env']} to use {provider}")
        client = OpenAI(api_key=os.environ.get(cfg["env"], "ollama") if cfg["env"] else "ollama",
                        base_url=cfg["base_url"])
        return instructor.from_openai(client, mode=getattr(instructor.Mode, cfg["mode"])), cfg["model"]

    def _build_engine(spec: Dict[str, Any]):
        from pydantic import create_model
        engine = AuraEngine()
        provider = spec.get("provider", "ollama")
        needs_llm = any(n.get("type") == "extract" for n in spec["nodes"])
        default_model = None
        if needs_llm:
            engine.client, default_model = _client_for(provider)
            engine.provider.register_client("default", engine.client)

        edgemap: Dict[str, List[str]] = {}
        for a, b in spec.get("edges", []):
            edgemap.setdefault(a, []).append(b)
        typemap = {"str": str, "int": int, "float": float, "bool": bool}

        def make_handle(target):
            def handle(self, user_text, extracted_data=None, memory=None):
                data = extracted_data.model_dump() if extracted_data is not None else dict(memory or {})
                return target, data
            return handle

        classes = {}
        for n in spec["nodes"]:
            nxt = edgemap.get(n["id"], [])
            attrs: Dict[str, Any] = {
                "system_prompt": n.get("system_prompt") or n["id"],
                "model": n.get("model") or default_model or "gpt-4o",
                "obligations": list(n.get("obligations", [])),
                "consensus": int(n.get("consensus", 1) or 1),
                "confidence": float(n.get("confidence", 0.9) or 0.9),
                "handle": make_handle(nxt[0] if nxt else "END"),
            }
            cap = n.get("capability", "plain")
            if cap == "untrusted": attrs["untrusted_source"] = True
            elif cap == "sink": attrs["dangerous_sink"] = True
            elif cap == "sanitizer": attrs["sanitizer"] = True
            if n.get("sandbox_rule"): attrs["sandbox_rule"] = n["sandbox_rule"]
            if n.get("type") == "extract":
                defs = {f["name"]: (typemap.get(f.get("type", "str"), str), ...)
                        for f in n.get("fields", []) if f.get("name")}
                if defs:
                    attrs["extracts"] = create_model(n["id"] + "Data", **defs)
            cls = type(n["id"], (Node,), attrs)
            classes[n["id"]] = cls
            engine.register(cls)
        for a, b in spec.get("edges", []):
            if a in classes and b in classes:
                engine.connect([CompiledTransition(from_node=classes[a], to_node=classes[b])])
        return engine

    class RunReq(BaseModel):
        nodes: List[Dict[str, Any]]
        edges: List[List[str]] = []
        entry: Optional[str] = None
        input: str = ""
        provider: str = "ollama"

    @app.post("/api/run")
    def run(req: RunReq):
        spec = req.model_dump()
        if not spec["nodes"]:
            return {"error": "add at least one node"}
        try:
            engine = _build_engine(spec)
        except Exception as e:
            return {"error": str(e)[:220]}
        entry = req.entry or spec["nodes"][0]["id"]
        state, memory, trace = entry, {}, []
        for _ in range(24):
            try:
                nxt, payload = engine.process(state, req.input, memory=memory)
            except Exception as e:
                trace.append({"node": state, "error": str(e)[:180]})
                break
            reps = engine.verification_reports()
            rep = reps[-1] if reps else {}
            step = {"node": state, "next": nxt,
                    "extracted": payload if isinstance(payload, dict) else {},
                    "verified": rep.get("extraction_verified", rep.get("contract_verified")),
                    "iterations": rep.get("iterations"),
                    "abstained": rep.get("abstained", False)}
            conf = rep.get("conformal")
            if conf is not None:
                step["conformal"] = {"covered": list(getattr(conf, "covered_fields", []))}
            trace.append(step)
            memory = payload if isinstance(payload, dict) else memory
            if nxt == "END" or nxt not in engine._nodes:
                break
            state = nxt
        try:
            contract = engine.compile_contract().model_dump()
        except Exception:
            contract = None
        return {"trace": trace, "contract": contract, "steps": len(trace)}

    # ── Flow persistence (save / load agents as JSON) ──
    _FLOW_DIR = os.path.join(os.path.expanduser("~"), ".aura_studio", "flows")

    class SaveReq(BaseModel):
        name: str
        flow: Dict[str, Any]

    @app.post("/api/flows/save")
    def save_flow(req: SaveReq):
        os.makedirs(_FLOW_DIR, exist_ok=True)
        safe = "".join(c for c in req.name if c.isalnum() or c in "-_ ").strip() or "agent"
        with open(os.path.join(_FLOW_DIR, safe + ".json"), "w") as f:
            json.dump(req.flow, f, indent=2)
        return {"ok": True, "name": safe}

    @app.get("/api/flows")
    def list_flows():
        if not os.path.isdir(_FLOW_DIR):
            return []
        return sorted(n[:-5] for n in os.listdir(_FLOW_DIR) if n.endswith(".json"))

    @app.get("/api/flows/{name}")
    def get_flow(name: str):
        safe = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()
        path = os.path.join(_FLOW_DIR, safe + ".json")
        if not os.path.isfile(path):
            return JSONResponse({"error": "not found"}, status_code=404)
        with open(path) as f:
            return json.load(f)

    if os.path.isdir(_STATIC):
        app.mount("/static", StaticFiles(directory=_STATIC), name="static")
    return app
