"""0016: `aura-state demo` (zero-setup showcase) + check exit codes + proof badge."""
import json

from aura_state.cli import main


def test_demo_runs_with_zero_setup_and_shows_finding_and_fix_fixes_0016(capsys):
    # No paths, no keys, no network — the demo agent is embedded in the package.
    code = main(["demo", "--no-color"])
    out = capsys.readouterr().out
    assert code == 1                                  # the demo agent is vulnerable
    assert "lethal trifecta" in out                   # the finding
    assert "sanitizer" in out                          # the fix guidance
    assert "checked:" in out and "trifecta" in out    # the proof badge line
    assert "never run" in out                          # honest: static, not executed


def test_demo_json_is_machine_readable_fixes_0016(capsys):
    code = main(["demo", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert code == 1 and data["verified"] is False
    agent = data["agents"][0]
    assert any(f["check"] == "trifecta" for f in agent["findings"])


def test_check_exit_codes_fixes_0016(tmp_path, capsys):
    # vulnerable → exit 1
    vuln = {"name": "v", "entry": "Ask", "edges": [["Ask", "Send"]],
            "nodes": [{"id": "Ask", "kind": "extract", "capability": "untrusted"},
                      {"id": "Send", "kind": "tool", "tool_name": "email.send", "side_effect": "external"}]}
    # proven → exit 0 (sanitizer between the untrusted source and the sink)
    clean = {"name": "c", "entry": "Ask", "edges": [["Ask", "Guard"], ["Guard", "Send"]],
             "nodes": [{"id": "Ask", "kind": "extract", "capability": "untrusted"},
                       {"id": "Guard", "kind": "sanitizer"},
                       {"id": "Send", "kind": "tool", "tool_name": "email.send", "side_effect": "external"}]}
    pv = tmp_path / "v.json"; pv.write_text(json.dumps(vuln))
    pc = tmp_path / "c.json"; pc.write_text(json.dumps(clean))
    assert main(["check", str(pv), "--no-color"]) == 1
    capsys.readouterr()
    assert main(["check", str(pc), "--no-color"]) == 0
    assert "PROVEN" in capsys.readouterr().out
