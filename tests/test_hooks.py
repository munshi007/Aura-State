"""SDK hooks: verify in your own code, fail-closed decorator."""
import pytest
from aura_state.hooks import verify, verified, VerificationError, Monitor


def test_verify_helper():
    assert verify({"a": 2, "b": 3, "t": 6}, ["t == a*b"]).verified is True
    assert verify({"a": 2, "b": 3, "t": 7}, ["t == a*b"]).verified is False


def test_verified_decorator_strict_raises():
    @verified(["total == area * rate"], strict=True)
    def bad():
        return {"area": 100, "rate": 3, "total": 999}

    with pytest.raises(VerificationError):
        bad()


def test_verified_decorator_passes_and_returns():
    @verified(["total == area * rate"], strict=True)
    def good():
        return {"area": 100, "rate": 3, "total": 300}

    assert good() == {"area": 100, "rate": 3, "total": 300}


def test_monitor_records_locally_when_studio_absent():
    # No studio running -> the network send is skipped, verification still happens.
    m = Monitor(url="http://127.0.0.1:59999", timeout=0.2)
    r = m.record({"a": 2, "b": 3, "t": 6}, ["t == a*b"], node="X")
    assert r.verified is True
