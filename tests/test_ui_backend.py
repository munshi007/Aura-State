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
