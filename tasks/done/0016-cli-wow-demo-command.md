# 0016: CLI time-to-wow — `aura-state demo` + screenshot-worthy `check` output

**Status:** done (2026-09-27)
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

- [x] `aura-state demo` runs with zero setup and shows a real finding + fix + proof badge
- [x] bundled example ships in the installed wheel (embedded in `aura_state/demo.py`, a package module — no `examples/` dependency; clean-room smoke deferred to 0017)
- [x] `check` output has finding → exact fix → badge; honors `NO_COLOR`/`--no-color`
- [x] CI exit codes verified (1 on blocking finding, 0 on proven)
- [x] tests `test_cli_*_fixes_0016`, all passing

## Notes

Demo agent: the realistic LangGraph support-ticket agent (fetch=untrusted, lookup=private DB, send=email exfil) — its lethal trifecta is caught. Source is **embedded** in `aura_state/demo.py` (not read from `examples/`) so it works from any clean install with no file paths/keys/network.

Exit-code contract: `check`/`demo` return `1` on a blocking (critical/high) finding, `0` when proven, `2` on load error; `--baseline` returns `1` only on NEW blocking findings.

## Completion (2026-09-27)
Added `aura_state/demo.py` (embedded LangGraph agent + `demo_flow()` via the static importer, never executed). Refactored `aura_state/cli.py`: extracted `_render(reports, args, baseline)` (shared by `check` + `demo`), added `_cmd_demo` and the `demo` subparser, and `_badge()` — an honest one-line proof badge printed for the single-agent case listing exactly which checks ran + each verdict (`checked: trifecta closed · taint proven · reachability proven · obligations proven · policy 0 flagged`). `demo` shows the finding + the fix guidance + "Try it on your own agent" footer, exits 1 (vulnerable), and supports `--json`/`--no-color`. Tests `tests/test_cli_demo_fixes_0016.py` (3). Full suite 238 passed.
Relates to [[0019]] (README shows this exact output) and [[0017]] (clean-room install verification).
