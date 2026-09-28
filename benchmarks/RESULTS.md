# aura-state benchmark

_Reproduce: `aura-state bench`_

```
  aura-state bench · design-time verification

  Accuracy on labeled synthetics (safe/vulnerable pairs, ground truth certain):
    case                      property    expected   detected   ok
    trifecta-open             trifecta    vuln       vuln       ✓
    trifecta-sanitized        trifecta    safe       safe       ✓
    taint-open                taint       vuln       vuln       ✓
    taint-sanitized           taint       safe       safe       ✓
    obligation-contradiction  obligation  vuln       vuln       ✓
    obligation-ok             obligation  safe       safe       ✓
    no-exfil-channel          trifecta    safe       safe       ✓
    clean-linear              clean       safe       safe       ✓

    recall 100%  ·  precision 100%  ·  silent misses 0 (sound)
    (recall is 100% by construction — fail-closed never silently passes a real finding;
     the measured number is precision: how often a flag is a true positive.)

  Import coverage on real-shaped agents (unmodified MCP/framework surfaces):
    autogpt_agent.tools.json      imported  6 nodes · findings · 3 finding(s)
    crewai_researcher.tools.json  imported  5 nodes · findings · 2 finding(s)
    db_reporter.mcp.json          imported  3 nodes · findings · 1 finding(s)
    github_triage.mcp.json        imported  4 nodes · findings · 2 finding(s)
    readonly_research.mcp.json    imported  4 nodes · proven · 0 finding(s)
    web_slack_assistant.mcp.json  imported  6 nodes · findings · 2 finding(s)

    import coverage: 100% (6/6 real agents ingested unmodified)
```
