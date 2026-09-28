# 0024: Compliance-grade verifiable certificate

**Status:** done (2026-09-28) — CLI certify/verify-cert; studio-cert migration deferred
**Type:** feature
**Tags:** `[compliance]` `[differentiator]` `[enterprise]`
**Priority:** later (Phase 1 — depth)
**Depends on:** 0015 (AuraContract / compile_contract — done), 0020, 0021
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** landscape research (2026-09-27). EU AI Act Art. 12 auditors want **evidence, not documentation**; banks/health/insurance demand governance proof before contracting (attributed). Proof-carrying/verifiable receipts are emerging (Microsoft Agent Governance Toolkit; PCAA arXiv:2606.04104) but are **runtime** ("Proof of Execution"). A **design-time, pre-deployment** verifiable certificate is the complementary artifact.

## Why

We already emit an `AuraContract` (0015). Upgrade it to a **machine-checkable, timestamped, content-addressable certificate** an external verifier or CI can validate **without trusting the operator** — aligned to Art. 12 / ISO 42001 evidence needs. This is the enterprise wedge and the same artifact the studio badge (0018) shows.

## What

- Extend `AuraContract` into a signed/hashed **certificate**: the proven properties (IFC non-interference, capability-containment, CTL, Z3 obligations, conformal), the graph hash, aura-state version, emit timestamp, and a **re-verification recipe** (an external checker can re-run the obligations/properties against the recorded design and confirm).
- `aura-state verify-cert <file>` — an independent verifier that re-checks the certificate's claims against the recorded design and reports match/mismatch.

## Approach

- Build on `aura_state/compiler/spec_compiler.py` (0015). Add signing/hashing + the re-verification entrypoint. Keep it standalone (no aura-runtime coupling).
- Map fields to Art. 12 / ISO 42001 evidence categories in docs; cite PCAA + MS receipts as the runtime counterpart we complement.

## Test strategy

- Round-trip + tamper: a modified certificate fails `verify-cert`.
- Re-verification: the certificate's obligations/properties re-check to the same verdicts against the recorded design.
- `test_certificate_*_fixes_0024`.

## Acceptance criteria

- [x] certificate = content-hashed design + verdict (full check_flow report) + standards mapping; re-verification recipe is `check_flow` on the recorded design (tamper-evidence via hashes + trustless re-verification; not crypto-signed — honest)
- [x] `aura-state certify` + `aura-state verify-cert` — independent re-verification; tamper on design (hash) OR verdict (hash + re-run) → INVALID, exit 1
- [x] `standards` field maps to EU AI Act Art. 12 / ISO 42001; module cites PCAA (2606.04104) + MS Agent Governance Toolkit receipts
- [~] complementary to the studio proof badge (0018) — both derive from the same verdict, but the badge is the visual artifact and this is the machine-verifiable one; unifying them is a follow-on
- [x] tests `tests/test_certificate_fixes_0024.py` (6)

## Notes

_record: certificate schema, signing scheme, the re-verification contract, evidence-category mapping._
Relates to [[0015]], [[0018]], [[0020]], [[0021]].

## Completion (2026-09-28)
Built `aura_state/certificate.py` — `make_certificate(flow, version, timestamp)` records the design + the `check_flow` verdict with content hashes (`design_hash`, `verdict_hash`) + a `standards` mapping (EU AI Act Art. 12, ISO/IEC 42001). `verify_certificate(cert)` re-checks it **without trusting the issuer**: recomputes both hashes (tamper-evidence) and — the real guarantee — **re-runs the verifier on the recorded design** and confirms the recorded verdict. CLI: `aura-state certify <agent> [--out]` and `aura-state verify-cert <file>`. Verified live: certify the vulnerable demo → honest "not proven" cert → `verify-cert` VALID; flip the claimed verdict → INVALID (verdict-hash mismatch + re-verification disagrees). 6 tests; full suite 262.

**Honest scope / deferred:** it is **content-hash + re-verification** tamper-evident, not cryptographically signed (no key) — the trustless property comes from anyone being able to re-run `check_flow` on the recorded design. The studio's existing `/api/certificate` still uses the pre-IFC engine taint and a different shape; migrating it onto `certificate.make_certificate` (so studio + CLI certs match, consistent with 0020) is a good follow-on. The 0018 badge and this cert both derive from the verdict but aren't yet one object.
