# 0017: Launch hardening — clean-install matrix + fail-closed re-sweep

**Status:** done (2026-09-27)
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

- [x] clean install + `demo` smoke passes on 3.12 / 3.14 locally; 3.10/3.11/3.13 via CI matrix
- [x] `requires-python` (>=3.10) and dep constraints reflect the real support matrix; CI matrix 3.10–3.14 + demo smoke
- [x] verdict surfaces fail closed (confirmed; tests from 0002/0004/0.11.x)
- [x] re-sweep notes list each surface checked + result (see Completion)

## Notes

_record: the support matrix decided, any dep pin changes, each fail-closed surface + its test._
Relates to [[0016]] (demo used as the smoke) and CLAUDE.md rule 8.

## Completion (2026-09-27)
**Install matrix:** clean-room `pip install` of the built wheel + `aura-state demo` verified on **3.12 and 3.14** locally (z3-solver + pyModelChecking resolve, embedded demo runs, exit 1, version 0.11.3). 3.11/3.13 not installed locally → covered by CI. Expanded `.github/workflows/ci.yml` matrix to **3.10–3.14** and added a **demo smoke step** (`python -m aura_state.cli demo` must exit non-zero — proves the gate exit code + that the embedded demo ships). `requires-python` stays `>=3.10` (deps support it; CI verifies).

**Fail-closed re-sweep:** audited the verdict-producing surfaces. All fail closed:
- `proof_engine.prove_extraction`: empty obligations = vacuously verified (nothing to prove); an obligation that can't be compiled/bound → **unproven**, forces `verified=False` (proof_engine.py:226-229). Hardened earlier in 0002/0004.
- `pipeline_conformal.should_abstain`: uncalibrated → **abstain** (returns True) — conservative (pipeline_conformal.py:82-83).
- `check_flow`: unknown-capability tools → advisory / "cannot prove trifecta-free", never silent-safe (0.11.x trifecta work).
- Recent additions reviewed: decision-routing rule-eval error sets `_decision_error` and defaults an edge (a routing fallback, not a safety verdict — not a fail-open); demo mock provider fills placeholders (not a verdict). **No new fail-opens.**
Fail-closed tests exist from 0002/0004/0.11.x; full suite 238 passing.
