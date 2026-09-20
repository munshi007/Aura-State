# We audited real MCP servers and agent frameworks for the lethal trifecta

**TL;DR — 5 of 6 common agent tool-compositions can close the [lethal trifecta](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/): untrusted content + private data + external comms on one reachable path, which is all a prompt injection needs to exfiltrate your data. Aura-State proves it statically, from a config, in one command. Reproduce every line yourself.**

```
$ python examples/audit/run.py

  target               trifecta  exfil sinks   real-world tie-in
  ──────────────────────────────────────────────────────────────────────────
  github_triage        CLOSED ✗  add_issue…    reproduces the Invariant Labs GitHub-MCP exploit
  web_slack_assistant  CLOSED ✗  slack_post…   common filesystem+fetch+slack desktop combo
  db_reporter          CLOSED ✗  slack_post…   postgres + web search + slack reporting agent
  readonly_research    broken ✓  —             read-only research agent — CONTRAST (safe)
  crewai_researcher    CLOSED ✗  CodeInterp…   CrewAI research crew tool surface
  autogpt_agent        CLOSED ✗  execute_py…   AutoGPT Classic command surface

  5/6 real compositions can close the lethal trifecta.
```

## What we are — and are not — claiming

This is **not** "project X has a bug." The lethal trifecta is an **emergent property of how tools are combined**, not a flaw in any single MCP server or framework — every tool here is behaving exactly as designed. The risk appears when you *wire them into one agent*, and nothing in today's tooling flags it at config time.

Every config in [`examples/audit/`](../examples/audit/) is modeled faithfully from a **real, published tool surface** (official MCP server READMEs / framework docs — sources in each file's `_comment`). Aura never connects to or runs a server; it reasons over the declared tools. Tool trust-classes are inferred from name + MCP annotations and are **override-able** per node — unknowns are surfaced as advisories, never silently assumed safe.

## The headline: this is a real, documented attack

The `github_triage` case reproduces the capability shape of the **Invariant Labs GitHub-MCP exploit** ([2025-05-26](https://invariantlabs.ai/blog/mcp-github-vulnerability)): a malicious **public** issue coerces an agent into reading a **private** repo and leaking it through a PR the attacker can read. Aura's verdict, from the config alone:

```
✗ trifecta [create_pull_request]: lethal trifecta closed — prompt injection at
  'get_issue' (untrusted) can reach external sink 'create_pull_request'
  unsanitized while 'get_file_contents' brings private data into scope.
  Path: get_issue → Agent → create_pull_request.
```

That is the exploit, caught **before** you connect the server — not after an incident.

## The same class, over and over

This pattern is behind a string of 2025 disclosures — all researcher PoCs / responsibly-disclosed, no confirmed in-the-wild criminal use:

| Incident | Mechanism | Source |
|---|---|---|
| **GitHub MCP** (Invariant Labs) | malicious issue → private repo read → leaked via public PR | [invariantlabs.ai](https://invariantlabs.ai/blog/mcp-github-vulnerability) |
| **EchoLeak** — M365 Copilot, CVE-2025-32711 | zero-click email injection → RAG executes it → M365 data exfiltrated | [thehackernews.com](https://thehackernews.com/2025/06/zero-click-ai-vulnerability-exposes.html) |
| **Slack MCP data leakage** (Embrace The Red) | injection → post a message whose URL unfurls, leaking data to the attacker | [embracethered.com](https://embracethered.com/blog/posts/2025/security-advisory-anthropic-slack-mcp-server-data-leakage/) |
| **MCP tool poisoning** (Invariant Labs) | malicious instructions hidden in a tool *description*; leaked SSH key / `mcp.json` | [invariantlabs.ai](https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks) |
| **WhatsApp MCP** (Invariant Labs) | "sleeper" server poisons a legit tool → leaks chat history + contacts | [invariantlabs.ai](https://invariantlabs.ai/blog/whatsapp-mcp-exploited) |

Different products, one shape: **untrusted content reaches an agent that can read private data and talk to the outside world.** That shape is exactly what Aura decides statically.

## Reproduce it

```bash
pip install aura-state            # or: uvx aura-state check ...
python examples/audit/run.py      # the whole table
uvx aura-state check examples/audit/github_triage.mcp.json   # one target, zero install
```

Point it at **your** setup: export your agent's MCP `tools/list` (or hand it your client config) and run `aura-state check your_tools.json`. If the trifecta closes, it tells you the three tools and the one sanitizer that breaks it.

## Method & honesty notes

- **Sound, may-reach:** if *some* path lets untrusted content reach an external sink unsanitized while private data is in scope, we report it. We over-flag rather than miss — correct for a security gate.
- **Classification is a heuristic** (name + MCP `readOnlyHint`/`openWorldHint`), override-able via `data_class` / `exfil`; unclassifiable read-tools become advisories, not false criticals. A read-only tool is never an exfil sink; a local `write_file` is a mutation, not the external-comms leg.
- **Design-time proof, not a runtime exploit.** Aura proves a reachability property over the tool graph; it does not execute anything. Runtime enforcement is a separate concern.
- Modeling caveat: only the filesystem server documents its annotation hints; other servers' read/act nature is classified from documented behavior. Tool names reflect the versions cited; some GitHub tools have since been renamed/consolidated.
