"""The `aura-state check --baseline` regression gate: fail only on NEW findings."""
import json

from aura_state.cli import main


CLEAN = {"name": "g", "entry": "Ask",
         "edges": [["Ask", "Guard"], ["Guard", "Send"]],
         "nodes": [{"id": "Ask", "kind": "extract", "capability": "untrusted"},
                   {"id": "Guard", "kind": "sanitizer"},
                   {"id": "Send", "kind": "tool", "tool_name": "email.send", "side_effect": "external"}]}

# same agent with the sanitizer removed — a real regression
REGRESSED = {"name": "g", "entry": "Ask",
             "edges": [["Ask", "Send"]],
             "nodes": [{"id": "Ask", "kind": "extract", "capability": "untrusted"},
                       {"id": "Send", "kind": "tool", "tool_name": "email.send", "side_effect": "external"}]}


def _write(p, obj):
    p.write_text(json.dumps(obj))
    return str(p)


def test_new_blocking_finding_fails_the_gate(tmp_path, capsys):
    base = tmp_path / "base.json"
    clean = _write(tmp_path / "clean.json", CLEAN)
    assert main(["check", clean, "--json"]) == 0
    base.write_text(capsys.readouterr().out)   # baseline = the clean report

    reg = _write(tmp_path / "reg.json", REGRESSED)
    assert main(["check", reg, "--baseline", str(base), "--no-color"]) == 1
    assert "REGRESSION" in capsys.readouterr().out


def test_preexisting_debt_does_not_fail_the_gate(tmp_path, capsys):
    reg = _write(tmp_path / "reg.json", REGRESSED)
    assert main(["check", reg, "--json"]) == 1     # blocking in absolute mode
    base = tmp_path / "base.json"
    base.write_text(capsys.readouterr().out)       # baseline already carries the debt

    # unchanged agent vs its own baseline: known, not new -> gate passes
    assert main(["check", reg, "--baseline", str(base), "--no-color"]) == 0
    assert "no new regressions" in capsys.readouterr().out


def test_absolute_mode_still_fails_without_baseline(tmp_path, capsys):
    reg = _write(tmp_path / "reg.json", REGRESSED)
    assert main(["check", reg, "--no-color"]) == 1
