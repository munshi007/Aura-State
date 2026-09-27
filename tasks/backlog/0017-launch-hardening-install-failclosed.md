# 0017: Launch hardening — clean-install matrix + fail-closed re-sweep

**Status:** backlog
**Type:** infra
**Tags:** `[install]` `[launch]` `[soundness]`
**Priority:** now (Phase 0 — launch)
**Depends on:** none
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** launch-readiness plan (2026-09-27). If `pip install` breaks on a common Python, virality dies at line 1. And any verdict that fails *open* destroys the credibility the whole product rests on.

## Why

Two launch-killers: (1) install friction on the Pythons people actually run (3.11/3.12/3.13, not just 3.14), and (2) a single fail-open verdict — the verifier is the product, so "it said safe but it wasn't" is fatal.

## What

- Verify a **clean-room** `pip install aura-state` on 3.11, 3.12, 3.13 (z3-solver, pyModelChecking, deps resolve and import).
- A focused re-sweep of the verification surfaces for any path that can return a passing/safe verdict on error (parse failure, unknown capability, solver `unknown`, too-few calibration samples) — must fail **closed** (unproven/advisory), never silently pass.

## Approach

- Clean-room per-version (fresh venv / container) install + `aura-state demo` smoke on each.
- Grep + read the verdict-producing paths (`check.py`, `verification/*`, loaders) for `except: pass`-style swallows and default-safe returns; each unknown must surface, not pass.
- Fix packaging (`requires-python`, dep pins) as needed for the matrix.

## Test strategy

- Add/confirm fail-closed unit tests at each surface: solver `unknown` → not verified; empty/malformed obligation → not verified; unknown capability → advisory not silent-safe; too-few conformal samples → uncalibrated, stated.
- CI job (or documented manual matrix) for the 3 Python versions.

## Acceptance criteria

- [ ] clean install + `demo` smoke passes on 3.11 / 3.12 / 3.13
- [ ] `requires-python` and dep constraints reflect the real support matrix
- [ ] every verdict surface has a fail-closed test (no path returns safe on error/unknown)
- [ ] re-sweep notes list each surface checked + result

## Notes

_record: the support matrix decided, any dep pin changes, each fail-closed surface + its test._
Relates to [[0016]] (demo used as the smoke) and CLAUDE.md rule 8.
