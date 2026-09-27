# 0016: CLI time-to-wow — `aura-state demo` + screenshot-worthy `check` output

**Status:** backlog
**Type:** feature
**Tags:** `[cli]` `[launch]` `[dx]`
**Priority:** now (Phase 0 — launch)
**Depends on:** none
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** launch-readiness plan (2026-09-27). The first thing an HN/PH visitor runs is the CLI; today's `check` output is clear but plain, and there is no zero-setup one-liner that shows the payoff. Time-to-wow is the single biggest conversion lever.

## Why

The viral screenshot is the terminal result. A visitor should run **one line**, no keys, no config, and see a real lethal-trifecta finding + the exact fix in ~3 seconds. Right now they must construct or find an agent file first, and the output isn't designed to be screenshotted.

## What

- `aura-state demo` — a subcommand that runs `check` on a **bundled** realistic vulnerable agent (reuse `examples/code/langgraph_stategraph.py`) with zero setup, and prints the finding + the one-line fix + a proof summary.
- Polish `aura-state check` output: the finding, the **exact fix** (sanitizer / capability to drop), and a proof badge line, with restrained color. Keep it honest (no fabricated confidence).
- Confirm CI exit codes: non-zero when a blocking finding exists, zero when proven. Documented for a CI gate.

## Approach

- Add the `demo` subcommand to the existing CLI (`aura_state/cli.py`), pointing at the bundled example; ensure the example ships in the wheel (`pyproject` package data / MANIFEST).
- Factor the `check` renderer so the finding → fix → badge block is one function; add `--no-color` and respect `NO_COLOR`.
- Verify exit codes end to end (`check` on a vulnerable agent → exit 1; on a clean one → exit 0).

## Test strategy

- `test_cli_demo_runs_without_setup_fixes_0016`: `demo` exits non-zero (finding present), prints the trifecta finding + a fix line, no network, no key.
- `test_cli_check_exit_codes_fixes_0016`: vulnerable → exit 1, clean → exit 0.
- Golden-ish assertion on the presence of the fix line + badge (not exact bytes).

## Acceptance criteria

- [ ] `aura-state demo` runs with zero setup and shows a real finding + fix + proof badge
- [ ] bundled example ships in the installed wheel (works from a clean venv, not just the repo)
- [ ] `check` output has finding → exact fix → badge; honors `NO_COLOR`/`--no-color`
- [ ] CI exit codes verified (1 on blocking finding, 0 on proven)
- [ ] tests `test_cli_*_fixes_0016`, all passing

## Notes

_record: the demo agent chosen, the wheel-packaging fix, exact exit-code contract._
Relates to [[0019]] (README shows this exact output) and [[0017]] (clean install).
