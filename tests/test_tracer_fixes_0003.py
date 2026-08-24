"""Regression tests for task 0003: no pickle in tracer, atomic JSON writes."""
import json
import os

import pytest

from aura_state.execution import tracer as tracer_mod
from aura_state.execution.tracer import AuraTrace, TraceFormatError


def test_tracer_no_pickle_fixes_0003(tmp_path):
    # pickle must not be imported anywhere in the tracer module.
    assert "pickle" not in dir(tracer_mod)
    src = os.path.join(os.path.dirname(tracer_mod.__file__), "tracer.py")
    with open(src) as f:
        text = f.read()
    assert "import pickle" not in text
    assert "pickle." not in text


def test_tracer_round_trip_json_fixes_0003(tmp_path):
    t = AuraTrace(trace_dir=str(tmp_path), session_id="sess")
    memory = {"budget": 450000, "city": "Seattle", "tags": ["a", "b"]}
    t.dump_node_state(step=1, node_name="ExtractLead", memory_context=memory, extracted=None)

    # Only JSON on disk -- no .pkl produced.
    files = os.listdir(os.path.join(str(tmp_path), "sess"))
    assert any(f.endswith(".json") for f in files)
    assert not any(f.endswith(".pkl") for f in files)

    loaded = AuraTrace.load_trace("sess", 1, trace_dir=str(tmp_path))
    assert loaded["node"] == "ExtractLead"
    assert loaded["memory"] == memory


def test_tracer_atomic_write_fixes_0003(tmp_path):
    # After a completed write no temp file remains and the live file is valid JSON.
    t = AuraTrace(trace_dir=str(tmp_path), session_id="sess")
    t.dump_node_state(step=2, node_name="Q", memory_context={"x": 1}, extracted=None)
    session_dir = os.path.join(str(tmp_path), "sess")
    assert not any(f.endswith(".tmp") for f in os.listdir(session_dir))
    path = os.path.join(session_dir, "step_002_Q.json")
    with open(path) as f:
        json.load(f)  # parses cleanly -> not torn


def test_tracer_rejects_malicious_pickle_fixes_0003(tmp_path):
    # A legacy/hostile .pkl-style byte blob dropped in as the step file must be
    # rejected on load, never deserialized as code.
    import pickle

    class Exploit:
        def __reduce__(self):
            return (os.system, ("echo pwned",))

    session_dir = os.path.join(str(tmp_path), "sess")
    os.makedirs(session_dir)
    # Write the exploit under the .json name the loader looks for.
    with open(os.path.join(session_dir, "step_003_x.json"), "wb") as f:
        f.write(pickle.dumps(Exploit()))

    with pytest.raises(TraceFormatError):
        AuraTrace.load_trace("sess", 3, trace_dir=str(tmp_path))


def test_tracer_rejects_wrong_schema_fixes_0003(tmp_path):
    session_dir = os.path.join(str(tmp_path), "sess")
    os.makedirs(session_dir)
    with open(os.path.join(session_dir, "step_004_x.json"), "w") as f:
        json.dump({"not": "a trace"}, f)
    with pytest.raises(TraceFormatError):
        AuraTrace.load_trace("sess", 4, trace_dir=str(tmp_path))
