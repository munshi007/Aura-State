# 0024: Compliance-grade verifiable certificate

**Status:** backlog
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

- [ ] certificate = signed/hashed contract + proven properties + re-verification recipe
- [ ] `aura-state verify-cert` independently re-checks and reports match/mismatch; tamper → fail
- [ ] docs map fields to EU AI Act Art. 12 / ISO 42001 evidence; cite PCAA (2606.04104) + MS receipts
- [ ] same artifact powers the studio proof badge (0018)
- [ ] tests `test_certificate_*_fixes_0024`

## Notes

_record: certificate schema, signing scheme, the re-verification contract, evidence-category mapping._
Relates to [[0015]], [[0018]], [[0020]], [[0021]].
