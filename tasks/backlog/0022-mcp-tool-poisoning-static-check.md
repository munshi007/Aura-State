# 0022: MCP tool-poisoning static check

**Status:** backlog
**Type:** feature
**Tags:** `[loaders]` `[security]` `[mcp]`
**Priority:** later (Phase 1 — depth)
**Depends on:** 0020 (IFC) helpful, not required
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** landscape research (2026-09-27). MCP supply-chain crisis: ~200k vulnerable instances; the tool-description field is an unsanitized injection surface; tool poisoning + config worms (CSA 2026, attributed). We already import MCP configs — a static poisoning check is a natural, timely add.

## Why

An MCP `tool` description can embed adversarial instructions the agent will follow ("ignore previous… send X to Y"), and a set of tools can jointly close the trifecta. We can flag both statically at import — before the agent ever connects to the server.

## What

- On MCP import, scan each tool's `description`/schema for **embedded-instruction / injection patterns** (imperative directives aimed at the agent, exfil verbs, credential/секрет references) → advisory finding on that tool.
- Reuse the trifecta/IFC pass to flag **capability combinations** across the imported tool surface that close a lethal path.

## Approach

- Extend the MCP loader (`aura_state/loaders/mcp.py`) with a description-scanner (pattern set for embedded instructions/exfil directives) producing advisory findings; keep it heuristic + honest (advisory, not a hard block).
- Feed the imported surface through IFC/trifecta for the combination risk (already the case; ensure descriptions are surfaced).
- Cite the MCP tool-poisoning research in the docstring.

## Test strategy

- A tool whose description contains an injected instruction ("ignore instructions and email …") → flagged.
- A benign description → not flagged.
- fetch+filesystem+slack surface → trifecta combination flagged (existing behavior, assert it holds).
- `test_mcp_poisoning_*_fixes_0022`.

## Acceptance criteria

- [ ] MCP import flags embedded-instruction/injection patterns in tool descriptions (advisory)
- [ ] capability-combination trifecta risk across the imported surface surfaced
- [ ] honest heuristic (advisory), no false hard-blocks; benign descriptions clean
- [ ] cites MCP tool-poisoning research; tests `test_mcp_poisoning_*_fixes_0022`

## Notes

_record: the pattern set, false-positive posture, examples used._
Relates to [[0020]], [[0023]].
