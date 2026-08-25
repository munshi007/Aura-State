"""
Hooks — verify your agent's output in your own code, and (optionally) stream it
to a running Aura Studio for a live, verified view.

This is how you use Aura-State inside an existing stack (CrewAI, LangGraph, plain
functions, a service): you don't rebuild your agent here — you verify its output
where it already runs.

    from aura_state.hooks import Monitor, verified

    mon = Monitor()                       # points at a local `aura-state ui`

    # 1) verify + stream any output
    mon.record({"refund_amount": 180, "order_total": 200},
               ["refund_amount <= order_total"], node="RefundAgent")

    # 2) decorate a function that returns a dict / pydantic model
    @verified(["total == area * rate"], monitor=mon, strict=True)
    def extract_quote(text) -> dict:
        ...   # your LLM call / CrewAI task / LangGraph node
        return {"area": 100, "rate": 3, "total": 300}

`strict=True` raises on an unproven output (fail-closed); otherwise it just
records the verdict and returns the value unchanged.
"""
from typing import Any, Dict, List, Optional, Sequence

from .verification.proof_engine import prove_extraction, ProofResult


class VerificationError(Exception):
    """Raised by @verified(strict=True) when an output can't be proven."""


def verify(data: Dict[str, Any], obligations: Sequence[str]) -> ProofResult:
    """Prove `data` against `obligations` with Z3 (no server needed)."""
    return prove_extraction(dict(data), list(obligations))


def _to_dict(out: Any) -> Dict[str, Any]:
    if hasattr(out, "model_dump"):
        return out.model_dump()
    if isinstance(out, dict):
        return out
    raise TypeError("verified() expects the wrapped function to return a dict or a pydantic model")


class Monitor:
    """A tiny client that verifies an output and streams it to a running studio.

    Non-fatal by design: if the studio isn't running, verification still happens
    locally and the network send is silently skipped.
    """

    def __init__(self, url: str = "http://127.0.0.1:8155", timeout: float = 1.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def record(self, data: Dict[str, Any], obligations: Sequence[str] = (),
               node: str = "extraction", source: Optional[str] = None) -> ProofResult:
        data = _to_dict(data) if not isinstance(data, dict) else dict(data)
        result = prove_extraction(data, list(obligations))
        try:
            import json
            import urllib.request
            body = json.dumps({"node": node, "source": source, "data": data,
                               "obligations": list(obligations)}).encode()
            req = urllib.request.Request(self.url + "/api/ingest", body,
                                         {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=self.timeout)
        except Exception:
            pass   # studio not running — verification still happened locally
        return result


def verified(obligations: Sequence[str], monitor: Optional[Monitor] = None,
             node: Optional[str] = None, strict: bool = False):
    """Decorator: verify a function's dict/pydantic output against `obligations`.

    Streams to `monitor` if given. With `strict=True`, raises VerificationError
    when the output isn't proven (fail-closed).
    """
    def deco(fn):
        def wrap(*args, **kwargs):
            out = fn(*args, **kwargs)
            data = _to_dict(out)
            name = node or getattr(fn, "__name__", "extraction")
            if monitor is not None:
                result = monitor.record(data, obligations, node=name)
            else:
                result = prove_extraction(data, list(obligations))
            if strict and not result.verified:
                raise VerificationError(
                    f"{name}: output not proven — failed {result.failed_obligations}, "
                    f"unproven {result.unproven_obligations}"
                )
            return out
        wrap.__name__ = getattr(fn, "__name__", "wrapped")
        wrap.__doc__ = getattr(fn, "__doc__", None)
        return wrap
    return deco
