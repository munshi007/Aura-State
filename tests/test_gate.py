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


# ── 0.7.2: --json honors --baseline, and finding_key discriminates by source ──

ONE_SOURCE = {"name": "g2", "entry": "A",
              "edges": [["A", "S"]],
              "nodes": [{"id": "A", "kind": "extract", "capability": "untrusted"},
                        {"id": "S", "kind": "tool", "tool_name": "http.post", "side_effect": "external"}]}
# same sink S, but a NEW untrusted source B also reaches it
TWO_SOURCES = {"name": "g2", "entry": "A",
               "edges": [["A", "S"], ["B", "S"]],
               "nodes": [{"id": "A", "kind": "extract", "capability": "untrusted"},
                         {"id": "B", "kind": "extract", "capability": "untrusted"},
                         {"id": "S", "kind": "tool", "tool_name": "http.post", "side_effect": "external"}]}


def test_json_mode_honors_baseline(tmp_path, capsys):
    one = _write(tmp_path / "one.json", ONE_SOURCE)
    assert main(["check", one, "--json"]) in (0, 1)
    base = tmp_path / "base.json"; base.write_text(capsys.readouterr().out)
    # unchanged agent vs its own baseline, JSON mode => no regression => exit 0
    assert main(["check", one, "--json", "--baseline", str(base)]) == 0
    assert '"mode": "regression"' in capsys.readouterr().out


def test_new_source_into_existing_sink_is_a_regression(tmp_path, capsys):
    one = _write(tmp_path / "one.json", ONE_SOURCE)
    assert main(["check", one, "--json"]) in (0, 1)
    base = tmp_path / "base.json"; base.write_text(capsys.readouterr().out)
    # a second untrusted source now reaches the same sink S -> NEW finding
    two = _write(tmp_path / "two.json", TWO_SOURCES)
    assert main(["check", two, "--baseline", str(base), "--no-color"]) == 1
    assert "REGRESSION" in capsys.readouterr().out
