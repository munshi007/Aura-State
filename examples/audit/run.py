"""Reproducible lethal-trifecta audit of real MCP-server compositions and agent
frameworks.

    python examples/audit/run.py
    # or, zero-install, per file:
    #   uvx aura-state check examples/audit/github_triage.mcp.json

Every config here is modeled faithfully from a real tool surface (official MCP
server READMEs / framework docs) — see each file's `_comment` for sources. We
audit how the tools *combine*, which is where the lethal trifecta lives; we do
NOT claim a bug in any single project, and we never connect to or run a server.
"""
import glob
import json
import logging
import os

logging.getLogger("aura_state").setLevel(logging.ERROR)
from aura_state.check import check_flow
from aura_state.loaders.mcp import is_mcp_config, flow_from_mcp

HERE = os.path.dirname(__file__)

# target -> (real-world tie-in, source URL). Order = display order.
TARGETS = {
    "github_triage":      ("reproduces the Invariant Labs GitHub-MCP exploit", "invariantlabs.ai/blog/mcp-github-vulnerability"),
    "web_slack_assistant": ("common filesystem+fetch+slack desktop combo",     "embracethered.com/blog/posts/2025/security-advisory-anthropic-slack-mcp-server-data-leakage"),
    "db_reporter":        ("postgres + web search + slack reporting agent",     "modelcontextprotocol/servers-archived"),
    "readonly_research":  ("read-only research agent — CONTRAST (safe)",        "modelcontextprotocol/servers"),
    "crewai_researcher":  ("CrewAI research crew tool surface",                 "docs.crewai.com/en/concepts/tools"),
    "autogpt_agent":      ("AutoGPT Classic command surface",                   "agpt.co/docs/classic"),
}


def _load(path):
    with open(path) as f:
        obj = json.load(f)
    return flow_from_mcp(obj, name=os.path.basename(path).split(".")[0]) if is_mcp_config(obj) else obj


def main() -> None:
    files = {os.path.basename(p).split(".")[0]: p for p in glob.glob(os.path.join(HERE, "*.json"))}
    rows = []
    for key, (tie, src) in TARGETS.items():
        p = files.get(key)
        if not p:
            continue
        r = check_flow(_load(p))
        tri = [f for f in r.findings if f.check == "trifecta" and f.severity == "critical"]
        rows.append((key, tie, src, len(tri), r))

    print("\n  Lethal-trifecta audit · real MCP servers + agent frameworks\n")
    w = max(len(k) for k, *_ in rows) + 2
    print(f"  {'target'.ljust(w)}{'trifecta'.ljust(10)}{'exfil sinks'.ljust(13)}real-world tie-in")
    print("  " + "─" * (w + 60))
    for key, tie, src, ntri, r in rows:
        verdict = "CLOSED ✗" if ntri else "broken ✓"
        sinks = ", ".join(sorted({f.node for f in r.findings if f.check == "trifecta" and f.severity == "critical"})) or "—"
        print(f"  {key.ljust(w)}{verdict.ljust(10)}{sinks[:11].ljust(13)}{tie}")

    closed = sum(1 for *_, ntri, _ in rows if ntri)
    print("\n  " + "─" * (w + 60))
    print(f"  {closed}/{len(rows)} real compositions can close the lethal trifecta "
          f"(untrusted content → private data → external comms, one reachable path).")
    print("  Reproduce any line:  uvx aura-state check examples/audit/<file>\n")
    print("  Sources per target:")
    for key, (tie, src) in TARGETS.items():
        print(f"    {key.ljust(w)} {src}")
    print()


if __name__ == "__main__":
    main()
