# 0018: Studio front door + shareable proof badge

**Status:** backlog
**Type:** feature
**Tags:** `[platform]` `[launch]` `[dx]` `[virality]`
**Priority:** now (Phase 0 — launch)
**Depends on:** none
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** launch-readiness plan (2026-09-27). The studio is a 13-tab tool — powerful but it makes a first-timer *work* to reach value ("feels rigid"). Viral products give the payoff in one move. And a verification today produces only a JSON certificate — nothing shareable.

## Why

The one move that sells: **import your agent → instant verdict + the finding on the graph**, then reveal the deeper tabs progressively. And a verified result should produce a **shareable, embeddable artifact** — built-in virality *and* the compliance/audit evidence enterprises ask for.

## What

- **Front door**: on empty/first load, lead with import-from-code / paste-agent → run verdict → finding lights the graph. The 13 modules are revealed after, not up front.
- **Shareable proof badge**: after a verify, generate a small PNG/SVG + embeddable snippet ("✓ Verified by aura-state — trifecta-free · terminates · calibrated", with the contract hash). Downloadable; links back.

## Approach

- Frontend (`frontend/src`): a first-run/empty state that foregrounds import→verdict; keep the rail but de-emphasize until an agent is loaded.
- Badge: render server-side or client-side from the verify result + contract hash; expose a download + copy-embed. Reuse the existing certificate content as the source of truth.
- No fabricated claims on the badge — only what was actually proven for that agent.

## Test strategy

- Frontend: the empty state shows the import affordance first (component/DOM test).
- Badge reflects the real verdict (proven vs violated) and the contract hash; a violated agent cannot render a "verified" badge.
- Drive the browser: import → verdict → badge present (Playwright, as in prior studio E2E).

## Acceptance criteria

- [ ] first-run studio leads with import→verdict; deeper tabs revealed progressively
- [ ] shareable proof badge (PNG/SVG + embed snippet) generated from a real verify result
- [ ] badge is honest — only proven properties; a vulnerable agent gets no "verified" badge
- [ ] browser E2E: import → verdict → badge

## Notes

_record: badge design + fields, where it's generated, the embed format._
Relates to [[0024]] (compliance certificate is the same artifact, formalized) and the "feels rigid" feedback.
