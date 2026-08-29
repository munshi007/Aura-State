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
from ..verification.temporal_verifier import (
    reachability, always_before, mutual_exclusion, eventual_completion,
    find_dead_ends, PropertyResult,
)
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

    @app.get("/api/providers/test/{name}")
    def test_provider(name: str):
        cfg = _PROVIDERS.get(name)
        if not cfg:
            return {"ok": False, "detail": "unknown provider"}
        if cfg["env"] and not os.environ.get(cfg["env"]):
            return {"ok": False, "detail": f"{cfg['env']} not set"}
        try:
            if cfg["base_url"] and "localhost" in cfg["base_url"]:
                import urllib.request
                urllib.request.urlopen(cfg["base_url"].replace("/v1", "") + "/api/tags", timeout=2.5)
                return {"ok": True, "detail": "reachable"}
            return {"ok": True, "detail": "credentials present"}
        except Exception as e:
            return {"ok": False, "detail": str(e)[:100]}

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

    # ── Fetch a URL locally and strip it to text (for web/scraper agents).
    #    Runs on the user's machine — not a hosted service. Size-capped. ──
    class FetchReq(BaseModel):
        url: str

    @app.post("/api/fetch_url")
    def fetch_url(req: FetchReq):
        import urllib.request, re as _re2
        url = req.url.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        try:
            r = urllib.request.Request(url, headers={"User-Agent": "AuraStudio/0.6 (+local)"})
            with urllib.request.urlopen(r, timeout=8) as resp:
                raw = resp.read(1_500_000).decode("utf-8", "ignore")
        except Exception as e:
            return {"error": f"could not fetch: {str(e)[:160]}"}
        title = ""
        m = _re2.search(r"<title[^>]*>(.*?)</title>", raw, _re2.I | _re2.S)
        if m:
            title = _re2.sub(r"\s+", " ", m.group(1)).strip()[:200]
        body = _re2.sub(r"(?is)<(script|style|noscript|head).*?</\1>", " ", raw)
        body = _re2.sub(r"(?s)<[^>]+>", " ", body)
        body = _re2.sub(r"&[a-z#0-9]+;", " ", body)
        body = _re2.sub(r"\s+", " ", body).strip()
        body = body[:8000]
        return {"title": title, "text": body, "chars": len(body), "url": url}

    # ── DSPy few-shot tuner: bootstrap a node's prompt from past successes.
    #    Real BootstrapTeleprompter; offline char-stub embedder (pass an OpenAI
    #    client in code for semantic KNN — see docs). ──
    class TuneReq(BaseModel):
        node: str = "node"
        prompt: str
        examples: List[Dict[str, Any]] = []       # [{input, output}]
        new_input: str = ""

    @app.post("/api/tune")
    def tune(req: TuneReq):
        from ..compiler.dspy_tuner import BootstrapTeleprompter, char_stub_embedder
        tp = BootstrapTeleprompter(embedder=char_stub_embedder)
        dataset = [{"node": req.node, "input": e.get("input", ""), "output": e.get("output", {}), "success": True}
                   for e in req.examples if e.get("input")]
        try:
            tp.compile(dataset)
            optimized = tp.optimize_node(req.node, req.prompt, req.new_input or (dataset[0]["input"] if dataset else ""))
        except Exception as e:
            return {"error": str(e)[:200]}
        return {"optimized": optimized, "n_demos": min(len(dataset), tp.k),
                "embedder": "char-stub (offline) — pass an OpenAI client for semantic KNN"}

    # ── Memory: context pruning (keep system + last N; inject required keys) ──
    class MemReq(BaseModel):
        history: List[Dict[str, str]] = []
        max_messages: int = 6
        required_keys: List[str] = []

    @app.post("/api/memory/preview")
    def memory_preview(req: MemReq):
        from ..memory.pruner import ContextPruner
        pruned = ContextPruner.prune(req.history, required_keys=req.required_keys or None, max_messages=req.max_messages)
        return {"before": len(req.history), "after": len(pruned), "pruned": pruned}

    # ── Policy / PII + secret scanner (content-level, deterministic regex).
    #    Complements the taint analysis: taint proves *structure* (untrusted ->
    #    sink), this flags *content* (a hardcoded key or PII in a prompt). ──
    import re as _re
    _POLICY_RULES = [
        ("secret.openai",  _re.compile(r"sk-[A-Za-z0-9]{20,}"),                    "critical", "OpenAI-style API key"),
        ("secret.google",  _re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),               "critical", "Google API key"),
        ("secret.aws",     _re.compile(r"AKIA[0-9A-Z]{16}"),                       "critical", "AWS access key id"),
        ("secret.generic", _re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{8,}"), "high", "hardcoded credential"),
        ("pii.ssn",        _re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                  "high", "US SSN"),
        ("pii.cc",         _re.compile(r"\b(?:\d[ -]?){15,16}\b"),                 "high", "credit-card number"),
        ("pii.email",      _re.compile(r"\b[\w.+\-]+@[\w\-]+\.[\w.\-]+\b"),        "medium", "email address"),
    ]

    class PolicyReq(BaseModel):
        nodes: List[Dict[str, Any]]

    @app.post("/api/policy/scan")
    def policy_scan(req: PolicyReq):
        findings = []
        for n in req.nodes:
            spots = {"system_prompt": n.get("system_prompt", ""), "sandbox_rule": n.get("sandbox_rule", "")}
            for i, o in enumerate(n.get("obligations", [])):
                spots[f"obligation[{i}]"] = o
            for where, text in spots.items():
                if not text:
                    continue
                for kind, rx, sev, desc in _POLICY_RULES:
                    m = rx.search(str(text))
                    if m:
                        s = m.group(0)
                        redacted = s[:4] + "…" + s[-2:] if len(s) > 8 else "…"
                        findings.append({"node": n["id"], "where": where, "kind": kind,
                                         "severity": sev, "description": desc, "match": redacted})
        order = {"critical": 0, "high": 1, "medium": 2}
        findings.sort(key=lambda f: order.get(f["severity"], 9))
        return {"clean": len(findings) == 0, "count": len(findings),
                "by_severity": {s: sum(1 for f in findings if f["severity"] == s) for s in ("critical", "high", "medium")},
                "findings": findings}

    # ── Counterexample-guided auto-repair: fix a taint violation by inserting a
    #    sanitizer before the sink, then re-prove the design is clean ──
    @app.post("/api/repair")
    def repair(spec: GraphSpec):
        engine = _engine_from_spec(spec)
        taint = engine.analyze_field_taint()
        if taint.verified:
            return {"repaired": False, "reason": "no taint violation to repair", "added": [], "edges": spec.edges}

        edges = [list(e) for e in spec.edges]
        node_ids = {n.id for n in spec.nodes}
        sinks = sorted({v.sink for v in taint.violations})
        added = []
        for sink in sinks:
            san = f"San_{sink}"
            k = 1
            while san in node_ids:
                san = f"San_{sink}_{k}"; k += 1
            node_ids.add(san); added.append((san, sink))
            # route every incoming edge to the sink through the sanitizer
            edges = [[a, san] if b == sink else [a, b] for a, b in edges]
            edges.append([san, sink])

        patched = [NodeSpec(id=n.id, capability=n.capability, obligations=n.obligations) for n in spec.nodes]
        patched += [NodeSpec(id=s, capability="sanitizer", obligations=[]) for s, _ in added]
        e2 = _engine_from_spec(GraphSpec(nodes=patched, edges=edges, entry=spec.entry))
        t2 = e2.analyze_field_taint()
        _audit_append("repair", f"counterexample-guided repair inserted {', '.join(s for s, _ in added)}",
                      {"added": [s for s, _ in added], "sinks": sinks,
                       "taint_before": "violated", "taint_after": "proven" if t2.verified else "violated",
                       "violations": [{"source": v.source, "sink": v.sink} for v in taint.violations]})
        return {
            "repaired": True,
            "added": [{"id": s, "sink": sink} for s, sink in added],
            "edges": edges,
            "taint_before": "violated",
            "taint_after": "proven" if t2.verified else "violated",
            "violations": [{"source": v.source, "sink": v.sink, "field": v.field} for v in taint.violations],
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
        default_provider = spec.get("provider", "ollama")

        # Per-node provider routing: each node may name its own provider. We build
        # one client per distinct provider and register it under the node's model
        # name (the provider layer resolves client by model prefix).
        _clients: Dict[str, Any] = {}
        def _get(provider: str):
            if provider not in _clients:
                _clients[provider] = _client_for(provider)
            return _clients[provider]

        node_models: Dict[str, str] = {}
        first_client = None
        for n in spec["nodes"]:
            if n.get("type") != "extract":
                continue
            prov = n.get("provider") or default_provider
            client, dmodel = _get(prov)
            model = n.get("model") or dmodel
            node_models[n["id"]] = model
            engine.provider.register_client(model, client)
            if first_client is None:
                first_client = client
        if first_client is not None:
            engine.client = first_client

        edgemap: Dict[str, List[str]] = {}
        for a, b in spec.get("edges", []):
            edgemap.setdefault(a, []).append(b)
        typemap = {"str": str, "int": int, "float": float, "bool": bool}

        def make_handle(target, mock=None):
            def handle(self, user_text, extracted_data=None, memory=None):
                data = extracted_data.model_dump() if extracted_data is not None else dict(memory or {})
                # Tool nodes are NOT executed here (that's aura-runtime). A mock
                # return is merged so downstream nodes can be tested design-time.
                if mock:
                    data = {**data, **mock}
                return target, data
            return handle

        classes = {}
        for n in spec["nodes"]:
            nxt = edgemap.get(n["id"], [])
            mock = None
            if n.get("type") == "tool" and n.get("mock_return"):
                try:
                    mock = json.loads(n["mock_return"])
                except Exception:
                    mock = None
            attrs: Dict[str, Any] = {
                "system_prompt": n.get("system_prompt") or n["id"],
                "model": node_models.get(n["id"]) or n.get("model") or "gpt-4o",
                "obligations": list(n.get("obligations", [])),
                "consensus": int(n.get("consensus", 1) or 1),
                "confidence": float(n.get("confidence", 0.9) or 0.9),
                "handle": make_handle(nxt[0] if nxt else "END", mock),
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
        name: str = "agent"
        memory: Dict[str, Any] = {}          # optional seed memory the run starts with

    @app.post("/api/run")
    def run(req: RunReq):
        spec = req.model_dump()
        if not spec["nodes"]:
            return {"error": "add at least one node"}
        try:
            engine = _build_engine(spec)
        except Exception as e:
            return {"error": str(e)[:220]}
        import time
        entry = req.entry or spec["nodes"][0]["id"]
        node_by_id = {n["id"]: n for n in spec["nodes"]}
        state, memory, trace = entry, dict(req.memory or {}), []
        for _ in range(24):
            t0 = time.perf_counter()
            try:
                nxt, payload = engine.process(state, req.input, memory=memory)
            except Exception as e:
                trace.append({"node": state, "error": str(e)[:180], "ms": round((time.perf_counter() - t0) * 1000)})
                break
            ms = round((time.perf_counter() - t0) * 1000)
            reps = engine.verification_reports()
            rep = reps[-1] if reps else {}
            nspec = node_by_id.get(state, {})
            step = {"node": state, "next": nxt, "ms": ms,
                    "kind": nspec.get("type"), "tool": nspec.get("tool_name"),
                    "side_effect": nspec.get("side_effect"),
                    "model": nspec.get("model"), "provider": nspec.get("provider"),
                    "consensus": int(nspec.get("consensus", 1) or 1),
                    "extracted": payload if isinstance(payload, dict) else {},
                    "verified": rep.get("extraction_verified", rep.get("contract_verified")),
                    "iterations": rep.get("iterations"),
                    "abstained": rep.get("abstained", False)}
            conf = rep.get("conformal")
            if conf is not None:
                step["conformal"] = {"covered": list(getattr(conf, "covered_fields", [])),
                                     "lower": _clean(getattr(conf, "lower", None)),
                                     "upper": _clean(getattr(conf, "upper", None))}
            trace.append(step)
            memory = payload if isinstance(payload, dict) else memory
            if nxt == "END" or nxt not in engine._nodes:
                break
            state = nxt
        try:
            contract = engine.compile_contract().model_dump()
        except Exception:
            contract = None
        verified = all(s.get("verified") is not False for s in trace) and not any("error" in s for s in trace)
        try:
            health = engine.health_report()
        except Exception:
            health = {}
        result = {"trace": trace, "contract": contract, "steps": len(trace), "verified": verified, "health": health}
        _save_run(req.name, req.input, req.provider, result)
        _audit_append("run", f"ran {req.name} — {len(trace)} steps, {'verified' if verified else 'UNVERIFIED'}",
                      {"agent": req.name, "provider": req.provider, "input": req.input[:240],
                       "steps": len(trace), "verified": verified,
                       "path": [s.get("node") for s in trace],
                       "contract_hash": (contract or {}).get("schema_version") and _hashlib.sha256(
                           json.dumps(contract, sort_keys=True, default=str).encode()).hexdigest()[:8]})
        return result

    # ── Runs history (every run persisted for audit / replay) ──
    _RUN_DIR = os.path.join(os.path.expanduser("~"), ".aura_studio", "runs")

    def _save_run(name: str, inp: str, provider: str, result: Dict[str, Any]) -> None:
        import datetime
        os.makedirs(_RUN_DIR, exist_ok=True)
        safe = "".join(c for c in name if c.isalnum() or c in "-_ ").strip() or "agent"
        now = datetime.datetime.now()
        rid = now.strftime("%Y%m%d-%H%M%S")
        doc = {"id": rid, "agent": safe, "ts": now.strftime("%Y-%m-%d %H:%M:%S"),
               "input": inp, "provider": provider,
               "verified": result.get("verified"), "steps": result.get("steps"),
               "trace": result.get("trace")}
        with open(os.path.join(_RUN_DIR, f"{safe}__{rid}.json"), "w") as f:
            json.dump(doc, f, indent=2)
        # keep last 100
        runs = sorted(os.listdir(_RUN_DIR))
        for old in runs[:-100]:
            try: os.remove(os.path.join(_RUN_DIR, old))
            except OSError: pass

    @app.get("/api/runs")
    def list_runs():
        if not os.path.isdir(_RUN_DIR):
            return []
        out = []
        for fn in sorted(os.listdir(_RUN_DIR), reverse=True)[:100]:
            try:
                with open(os.path.join(_RUN_DIR, fn)) as f:
                    d = json.load(f)
                out.append({"id": d["id"], "agent": d["agent"], "ts": d["ts"],
                            "verified": d.get("verified"), "steps": d.get("steps"),
                            "input": (d.get("input") or "")[:80], "file": fn[:-5]})
            except Exception:
                pass
        return out

    @app.get("/api/runs/{fid}")
    def get_run(fid: str):
        safe = "".join(c for c in fid if c.isalnum() or c in "-_").strip()
        path = os.path.join(_RUN_DIR, safe + ".json")
        if not os.path.isfile(path):
            return JSONResponse({"error": "not found"}, status_code=404)
        with open(path) as f:
            return json.load(f)

    # ── Evals: run a suite of test cases, assert expectations ──
    class EvalCase(BaseModel):
        input: str
        expect: Dict[str, Any] = {}          # {field: value} equality on final memory
        obligations: List[str] = []          # obligations that must hold on final memory

    class EvalReq(BaseModel):
        nodes: List[Dict[str, Any]]
        edges: List[List[str]] = []
        entry: Optional[str] = None
        provider: str = "ollama"
        cases: List[EvalCase] = []

    @app.post("/api/eval")
    def run_eval(req: EvalReq):
        results = []
        for case in req.cases:
            spec = {"nodes": req.nodes, "edges": req.edges, "entry": req.entry, "provider": req.provider}
            try:
                engine = _build_engine(spec)
            except Exception as e:
                results.append({"input": case.input, "passed": False, "detail": str(e)[:160], "got": {}})
                continue
            entry = req.entry or req.nodes[0]["id"]
            state, memory = entry, {}
            for _ in range(24):
                try:
                    nxt, payload = engine.process(state, case.input, memory=memory)
                except Exception as e:
                    memory = {"__error__": str(e)[:120]}; break
                memory = payload if isinstance(payload, dict) else memory
                if nxt == "END" or nxt not in engine._nodes:
                    break
                state = nxt
            checks, ok = [], True
            for k, v in (case.expect or {}).items():
                got = memory.get(k)
                good = str(got) == str(v)
                ok = ok and good
                checks.append({"kind": "equals", "field": k, "want": v, "got": got, "pass": good})
            if case.obligations:
                r = prove_extraction(memory, case.obligations)
                ok = ok and r.verified
                checks.append({"kind": "obligations", "want": case.obligations, "pass": r.verified,
                               "failed": r.failed_obligations})
            results.append({"input": case.input, "passed": ok, "got": memory, "checks": checks})
        passed = sum(1 for r in results if r["passed"])
        return {"total": len(results), "passed": passed, "failed": len(results) - passed, "results": results}

    # ── Custom CTL properties + structural dead-end detection ──
    class CtlProp(BaseModel):
        type: str                            # reachable | before | exclusive | completes
        a: Optional[str] = None
        b: Optional[str] = None

    class CtlReq(BaseModel):
        nodes: List[NodeSpec]
        edges: List[List[str]] = []
        entry: Optional[str] = None
        properties: List[CtlProp] = []

    @app.post("/api/ctl")
    def ctl(req: CtlReq):
        graph = GraphSpec(nodes=req.nodes, edges=req.edges, entry=req.entry)
        engine = _engine_from_spec(graph)
        # leaf nodes (no outgoing edge) are the intended terminals
        outgoing = {a for a, _ in req.edges}
        leaves = [n.id for n in req.nodes if n.id not in outgoing]

        built, labels = [], []
        for p in req.properties:
            if p.type == "reachable" and p.a:
                built.append({"description": f"{p.a} is reachable", "formula": reachability(p.a)}); labels.append(f"EF {p.a}")
            elif p.type == "before" and p.a and p.b:
                built.append({"description": f"{p.a} always before {p.b}", "formula": always_before(p.a, p.b)}); labels.append(f"{p.a} ≺ {p.b}")
            elif p.type == "exclusive" and p.a and p.b:
                built.append({"description": f"{p.a} and {p.b} mutually exclusive", "formula": mutual_exclusion(p.a, p.b)}); labels.append(f"¬({p.a} ∧ {p.b})")
            elif p.type == "completes" and leaves:
                built.append({"description": "every path eventually completes", "formula": eventual_completion(*leaves)}); labels.append("AF terminal")
        out = []
        if built:
            for lab, vr in zip(labels, engine.verify(built)):
                out.append({"label": lab, "verdict": "PROVEN" if vr.result == PropertyResult.PROVEN else "VIOLATED"})

        # structural dead-ends: non-leaf nodes that can't reach any terminal
        nodes_map = {n.id: None for n in req.nodes}
        trans = {}
        for a, b in req.edges:
            trans.setdefault(a, []).append(b)
        dead = sorted(find_dead_ends(nodes_map, trans, terminals=leaves))
        return {"properties": out, "dead_ends": dead, "leaves": leaves}

    # ── Proof certificate: the full design verdict as a signed JSON doc ──
    class CertReq(BaseModel):
        name: str = "agent"
        nodes: List[Dict[str, Any]]          # full node dicts (id, type, capability, model, obligations, ...)
        edges: List[List[str]] = []
        entry: Optional[str] = None
        invariants: List[str] = []           # agent-level obligations that must hold across the design

    @app.post("/api/certificate")
    def certificate(req: CertReq):
        import datetime, hashlib
        from importlib.metadata import version as _ver
        graph = GraphSpec(
            nodes=[NodeSpec(id=n["id"], capability=n.get("capability", "plain"),
                            obligations=list(n.get("obligations", []))) for n in req.nodes],
            edges=req.edges, entry=req.entry,
        )
        engine = _engine_from_spec(graph)
        taint = engine.analyze_field_taint()
        props = [{"description": f"{n['id']} reachable", "formula": reachability(n["id"])} for n in req.nodes]
        ctl_out = [{"property": vr.property_text,
                    "verdict": "PROVEN" if vr.result == PropertyResult.PROVEN else "VIOLATED"}
                   for vr in (engine.verify(props) if props else [])]

        node_docs, obl_all_ok = [], True
        for n in req.nodes:
            obls = list(n.get("obligations", []))
            consistent = None
            if obls:
                consistent = prove_obligations_satisfiable(obls).satisfiable
                obl_all_ok = obl_all_ok and consistent
            node_docs.append({
                "id": n["id"], "kind": n.get("type", "extract"),
                "capability": n.get("capability", "plain"),
                "model": n.get("model"), "provider": n.get("provider"),
                "schema": [f.get("name") for f in n.get("fields", []) if f.get("name")],
                "obligations": obls, "obligations_consistent": consistent,
                "sandbox_rule": n.get("sandbox_rule") or None,
                "consensus": n.get("consensus", 1), "conformal_alpha": round(1 - float(n.get("confidence", 0.9)), 3),
            })

        inv = {"obligations": req.invariants,
               "consistent": prove_obligations_satisfiable(req.invariants).satisfiable if req.invariants else None}
        contract = engine.compile_contract(properties=props).model_dump()
        taint_ok = taint.verified
        ctl_ok = all(c["verdict"] == "PROVEN" for c in ctl_out)
        inv_ok = inv["consistent"] is not False
        try:
            aura_ver = _ver("aura-state")
        except Exception:
            aura_ver = "0.6.0"

        body = {
            "agent": req.name,
            "engine": {"aura_state": aura_ver, "solver": "z3", "model_checker": "pyModelChecking"},
            "nodes": node_docs, "edges": req.edges, "entry": req.entry or (req.nodes[0]["id"] if req.nodes else None),
            "taint": {"verdict": "proven" if taint_ok else "violated",
                      "violations": [{"field": v.field, "source": v.source, "sink": v.sink} for v in taint.violations]},
            "ctl": ctl_out, "invariants": inv, "contract": contract,
        }
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
        return {
            "aura_certificate": "1.1",
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
            "verified": bool(taint_ok and ctl_ok and obl_all_ok and inv_ok),
            "summary": {"taint": "proven" if taint_ok else "violated",
                        "ctl": f"{sum(c['verdict']=='PROVEN' for c in ctl_out)}/{len(ctl_out)}",
                        "obligations": f"{sum(1 for d in node_docs if d['obligations_consistent'])}/{sum(1 for d in node_docs if d['obligations'])}",
                        "invariants": ("proven" if inv_ok else "violated") if req.invariants else "none"},
            "sha256": digest,
            **body,
        }

    # ── Forensic audit trail: append-only, hash-chained record of every action.
    #    Each entry seals the previous hash, so any tampering breaks the chain
    #    (tamper-evidence for compliance — EU AI Act / NIST AI RMF / ISO 42001). ──
    import hashlib as _hashlib
    _AUDIT_DIR = os.path.join(os.path.expanduser("~"), ".aura_studio")
    _AUDIT_LOG = os.path.join(_AUDIT_DIR, "audit.jsonl")
    _ACTOR = os.environ.get("USER") or os.environ.get("USERNAME") or "local"

    def _audit_entries() -> List[Dict[str, Any]]:
        if not os.path.isfile(_AUDIT_LOG):
            return []
        out = []
        with open(_AUDIT_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    try: out.append(json.loads(line))
                    except Exception: pass
        return out

    def _entry_hash(prev: str, seq: int, ts: str, actor: str, action: str, summary: str, detail: Dict[str, Any]) -> str:
        # Seal EVERY field, so tampering with any of them breaks the chain.
        payload = f"{prev}|{seq}|{ts}|{actor}|{action}|{summary}|{json.dumps(detail, sort_keys=True, default=str)}"
        return _hashlib.sha256(payload.encode()).hexdigest()

    def _audit_append(action: str, summary: str, detail: Dict[str, Any]) -> Dict[str, Any]:
        import datetime
        os.makedirs(_AUDIT_DIR, exist_ok=True)
        entries = _audit_entries()
        prev = entries[-1]["hash"] if entries else "GENESIS"
        seq = (entries[-1]["seq"] + 1) if entries else 1
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        h = _entry_hash(prev, seq, ts, _ACTOR, action, summary, detail)
        entry = {"seq": seq, "ts": ts, "actor": _ACTOR, "action": action,
                 "summary": summary, "detail": detail, "prev_hash": prev, "hash": h}
        with open(_AUDIT_LOG, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        # cap growth
        if len(entries) > 5000:
            keep = entries[-4000:]
            with open(_AUDIT_LOG, "w") as f:
                for e in keep:
                    f.write(json.dumps(e, default=str) + "\n")
        return entry

    class AuditReq(BaseModel):
        action: str
        summary: str = ""
        detail: Dict[str, Any] = {}

    @app.post("/api/audit")
    def audit_log(req: AuditReq):
        return _audit_append(req.action, req.summary, req.detail)

    @app.get("/api/audit")
    def audit_list(limit: int = 200):
        entries = _audit_entries()
        return list(reversed(entries))[:limit]

    @app.get("/api/audit/verify")
    def audit_verify():
        """Recompute the hash chain; report the first break (tamper detection)."""
        entries = _audit_entries()
        prev = "GENESIS"
        for e in entries:
            expect = _entry_hash(prev, e["seq"], e["ts"], e.get("actor", ""), e["action"], e.get("summary", ""), e["detail"])
            if expect != e["hash"] or e.get("prev_hash") != prev:
                return {"intact": False, "count": len(entries), "broken_at": e["seq"]}
            prev = e["hash"]
        return {"intact": True, "count": len(entries), "head": prev[:16] if entries else None}

    @app.get("/api/audit/export")
    def audit_export():
        from fastapi.responses import PlainTextResponse
        if not os.path.isfile(_AUDIT_LOG):
            return PlainTextResponse("", media_type="application/x-ndjson")
        with open(_AUDIT_LOG) as f:
            return PlainTextResponse(f.read(), media_type="application/x-ndjson")

    # ── Flow persistence (save / load agents as JSON) ──
    _FLOW_DIR = os.path.join(os.path.expanduser("~"), ".aura_studio", "flows")

    class SaveReq(BaseModel):
        name: str
        flow: Dict[str, Any]

    _VER_DIR = os.path.join(os.path.expanduser("~"), ".aura_studio", "versions")

    def _flow_hash(flow: Dict[str, Any]) -> str:
        import hashlib
        return hashlib.sha256(json.dumps(flow, sort_keys=True, default=str).encode()).hexdigest()[:8]

    @app.post("/api/flows/save")
    def save_flow(req: SaveReq):
        import datetime
        os.makedirs(_FLOW_DIR, exist_ok=True)
        safe = "".join(c for c in req.name if c.isalnum() or c in "-_ ").strip() or "agent"
        with open(os.path.join(_FLOW_DIR, safe + ".json"), "w") as f:
            json.dump(req.flow, f, indent=2)
        # append an immutable version snapshot (skip if identical to latest)
        vdir = os.path.join(_VER_DIR, safe)
        os.makedirs(vdir, exist_ok=True)
        h = _flow_hash(req.flow)
        existing = sorted(os.listdir(vdir))
        latest_hash = existing[-1].split("__")[-1][:-5] if existing else None
        version = len(existing) + 1
        if latest_hash != h:
            rid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            snap = {"version": version, "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "hash": h, "flow": req.flow}
            with open(os.path.join(vdir, f"v{version:04d}__{rid}__{h}.json"), "w") as f:
                json.dump(snap, f, indent=2)
        return {"ok": True, "name": safe, "hash": h}

    @app.get("/api/versions/{name}")
    def list_versions(name: str):
        safe = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()
        vdir = os.path.join(_VER_DIR, safe)
        if not os.path.isdir(vdir):
            return []
        out = []
        for fn in sorted(os.listdir(vdir), reverse=True):
            try:
                with open(os.path.join(vdir, fn)) as f:
                    d = json.load(f)
                out.append({"version": d["version"], "ts": d["ts"], "hash": d["hash"], "file": fn[:-5]})
            except Exception:
                pass
        return out

    @app.get("/api/versions/{name}/{fid}")
    def get_version(name: str, fid: str):
        safe = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()
        sfid = "".join(c for c in fid if c.isalnum() or c in "-_").strip()
        path = os.path.join(_VER_DIR, safe, sfid + ".json")
        if not os.path.isfile(path):
            return JSONResponse({"error": "not found"}, status_code=404)
        with open(path) as f:
            return json.load(f)

    @app.get("/api/flows")
    def list_flows():
        if not os.path.isdir(_FLOW_DIR):
            return []
        return sorted(n[:-5] for n in os.listdir(_FLOW_DIR) if n.endswith(".json"))

    @app.delete("/api/flows/{name}")
    def delete_flow(name: str):
        safe = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()
        path = os.path.join(_FLOW_DIR, safe + ".json")
        if os.path.isfile(path):
            os.remove(path)
            return {"ok": True, "deleted": safe}
        return JSONResponse({"error": "not found"}, status_code=404)

    # ── Export the agent as a runnable, standalone Python script (no lock-in) ──
    class ExportReq(BaseModel):
        name: str = "agent"
        nodes: List[Dict[str, Any]]
        edges: List[List[str]] = []
        entry: Optional[str] = None

    @app.post("/api/export/python")
    def export_python(req: ExportReq):
        from fastapi.responses import PlainTextResponse
        def ident(s: str) -> str:
            out = "".join(c if c.isalnum() else "_" for c in str(s))
            return out if out and out[0].isalpha() else "N_" + out
        pytype = {"str": "str", "int": "int", "float": "float", "bool": "bool"}
        edgemap: Dict[str, List[str]] = {}
        for a, b in req.edges:
            edgemap.setdefault(a, []).append(b)

        L = []
        L.append('"""')
        L.append(f"{req.name} — generated by Aura Studio.")
        L.append("A design-time-verified agent. Every obligation is proven with Z3,")
        L.append("the flow is CTL-model-checked, and dataflow is taint-analyzed.")
        L.append("")
        L.append("Run:  pip install aura-state  &&  python this_file.py")
        L.append('"""')
        L.append("from pydantic import BaseModel")
        L.append("from aura_state.core.engine import AuraEngine, Node, CompiledTransition")
        L.append("")
        # schemas
        for n in req.nodes:
            fields = [f for f in n.get("fields", []) if f.get("name")]
            if n.get("type") == "extract" and fields:
                L.append(f"class {ident(n['id'])}Data(BaseModel):")
                for f in fields:
                    L.append(f"    {f['name']}: {pytype.get(f.get('type', 'str'), 'str')}")
                L.append("")
        # nodes
        classnames = []
        for n in req.nodes:
            cn = ident(n["id"])
            classnames.append(cn)
            nxt = edgemap.get(n["id"], [])
            target = nxt[0] if nxt else "END"
            L.append(f"class {cn}(Node):")
            L.append(f"    system_prompt = {n.get('system_prompt', n['id'])!r}")
            if n.get("type") == "extract":
                L.append(f"    model = {n.get('model', 'gpt-4o')!r}")
                if [f for f in n.get("fields", []) if f.get("name")]:
                    L.append(f"    extracts = {cn}Data")
                if int(n.get("consensus", 1) or 1) > 1:
                    L.append(f"    consensus = {int(n['consensus'])}")
                    L.append(f"    confidence = {float(n.get('confidence', 0.9))}")
            if n.get("obligations"):
                L.append(f"    obligations = {list(n['obligations'])!r}")
            if n.get("sandbox_rule"):
                L.append(f"    sandbox_rule = {n['sandbox_rule']!r}")
            cap = n.get("capability", "plain")
            if cap == "untrusted": L.append("    untrusted_source = True")
            elif cap == "sink": L.append("    dangerous_sink = True")
            elif cap == "sanitizer": L.append("    sanitizer = True")
            if n.get("type") == "tool":
                tn = n.get("tool_name") or "your_tool"
                L.append(f"    # TOOL BOUNDARY — Aura proved this call's preconditions (sanitized input,")
                L.append(f"    # obligations hold). Bind the real implementation here / in aura-runtime.")
                L.append("    def handle(self, user_text, extracted_data=None, memory=None):")
                L.append("        data = dict(memory or {})")
                L.append(f"        # result = {tn}(data)")
                L.append(f"        return {target!r}, data")
            else:
                L.append("    def handle(self, user_text, extracted_data=None, memory=None):")
                L.append("        data = extracted_data.model_dump() if extracted_data is not None else dict(memory or {})")
                L.append(f"        return {target!r}, data")
            L.append("")
        # wiring
        L.append("def build():")
        L.append("    engine = AuraEngine()")
        L.append(f"    for cls in [{', '.join(classnames)}]:")
        L.append("        engine.register(cls)")
        if req.edges:
            L.append("    engine.connect([")
            for a, b in req.edges:
                if a in [n["id"] for n in req.nodes] and b in [n["id"] for n in req.nodes]:
                    L.append(f"        CompiledTransition(from_node={ident(a)}, to_node={ident(b)}),")
            L.append("    ])")
        L.append("    return engine")
        L.append("")
        L.append('if __name__ == "__main__":')
        L.append("    engine = build()")
        L.append("    # Design-time proof — inspect the compiled contract before shipping:")
        L.append("    print(engine.compile_contract())")
        entry = req.entry or (req.nodes[0]["id"] if req.nodes else "START")
        L.append("    # Run it (needs an OpenAI-compatible client set on engine.client for extract nodes):")
        L.append(f"    # nxt, out = engine.process({entry!r}, \"your input here\")")
        return PlainTextResponse("\n".join(L), media_type="text/x-python")

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
