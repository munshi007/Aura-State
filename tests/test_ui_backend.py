"""Aura Studio backend — real verifiers over a posted graph. Skipped if the
optional `ui` extra (fastapi) isn't installed."""
import pytest
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from aura_state.ui.server import create_app

client = TestClient(create_app())


def test_index_served():
    assert client.get("/").status_code == 200


def test_unsafe_graph_is_flagged():
    spec = {
        "nodes": [
            {"id": "Ingest", "capability": "untrusted", "obligations": []},
            {"id": "SendEmail", "capability": "sink", "obligations": []},
            {"id": "Price", "capability": "plain", "obligations": ["x > 5", "x < 3"]},
        ],
        "edges": [["Ingest", "SendEmail"], ["Ingest", "Price"]],
    }
    r = client.post("/api/verify", json=spec).json()
    assert r["taint"]["verdict"] == "VIOLATED"
    assert r["taint"]["violations"][0]["sink"] == "SendEmail"
    assert any(o["node"] == "Price" and o["consistent"] is False for o in r["obligations"])
    assert r["contract"]["taint"]["verdict"] == "VIOLATED"
    assert len(r["contract"]["meta"]["content_hash"]) == 64


def test_sanitizer_makes_it_safe():
    spec = {
        "nodes": [
            {"id": "Ingest", "capability": "untrusted", "obligations": []},
            {"id": "Review", "capability": "sanitizer", "obligations": []},
            {"id": "SendEmail", "capability": "sink", "obligations": []},
        ],
        "edges": [["Ingest", "Review"], ["Review", "SendEmail"]],
    }
    r = client.post("/api/verify", json=spec).json()
    assert r["taint"]["verdict"] == "PROVEN"


def test_conformal_endpoint():
    r = client.post("/api/conformal", json={"values": [100, 102, 98, 101, 99] * 5, "confidence": 0.9}).json()
    assert r["mode"] == "interval" and r["calibrated"] is True and r["lower"] <= r["upper"]


def test_risk_endpoint():
    import random
    rng = random.Random(0)
    s = [rng.random() for _ in range(300)]
    c = [rng.random() < x for x in s]
    r = client.post("/api/risk", json={"scores": s, "correct": c, "epsilon": 0.1, "test_score": 0.95}).json()
    assert r["calibrated"] is True
    assert r["realized_false_action_rate"] <= 0.13
    assert r["decision"] == "act"


def test_providers_endpoint():
    provs = {p["name"]: p for p in client.get("/api/providers").json()}
    assert "ollama" in provs and provs["ollama"]["available"] is True   # local, no key
    assert provs["openai"]["needs"] == "OPENAI_API_KEY"


def test_prove_endpoint():
    r = client.post("/api/prove", json={"data": {"area": 100, "rate": 3, "total": 999},
                                        "obligations": ["total == area * rate"]}).json()
    assert r["verified"] is False and "total == area * rate" in r["failed"]
    r2 = client.post("/api/prove", json={"data": {"x": 1}, "obligations": ["x > 5", "x < 3"]}).json()
    assert r2["consistent"] is False
