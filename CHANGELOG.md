# Changelog

All notable changes to Aura-State. Format loosely follows Keep a Changelog.

## [0.7.0]

### Added
- **Lethal-trifecta analysis** — the flagship. Statically decides whether an agent can close Simon Willison's *lethal trifecta* (private-data access + untrusted-content exposure + external communication on one reachable path → a prompt injection can exfiltrate). Reuses the taint/provenance engine for the injection leg and a role classifier for the private/exfil legs; reports the exact three nodes and the path. Surfaced in `aura-state check` as a critical finding and in `check_flow`'s summary. Tool roles are inferred from name/side-effect and are **override-able** per node (`data_class`, `exfil`); unclassifiable read-tools are surfaced as advisories, never dropped (fail-closed).
- **MCP importer** — `aura-state check` now auto-detects an MCP tool surface (`tools/list` result, a client config with tool lists, or a bare tool array) and models the worst case: an LLM planner hub that can call any tool in any order. Answers "do `fetch` + `filesystem` + `slack` together form an exfiltration channel?" statically — it never connects to or runs a server. `aura_state.loaders.mcp.flow_from_mcp`.
- **Regression gate** — `aura-state check --baseline <prior --json>` fails the build only on **NEW** blocking findings (marking each `[NEW]` / `[known]`), so a legacy agent's existing debt doesn't block PRs while any newly-opened injection/trifecta path does. The GitHub Action takes an optional `baseline` input.
- Two new example agents (`support_copilot`, `pr_triage_bot`) that exhibit the full trifecta, and an MCP example (`examples/mcp/`). `examples/audit.py` now reports the trifecta count (3 of 9 patterns close it).

## [0.6.0]

### Added
- **`aura-state check`** — a CI linter for agent designs. Statically verifies a flow (Z3 obligation consistency, CTL reachability/completion, taint dataflow, secret/PII scan) and exits non-zero on any blocking finding. Single or multi-file, `--json` output. Plus a composite **GitHub Action** (`action.yml`). ("mypy for AI agents.")
- **First-class Tool nodes** — a declared external call (name, side-effect: read/write/external, mock return). Aura proves the call's preconditions; it does not execute it. Run shows tool nodes as proven **boundaries** with a mock, not fake extractions. Export-to-Python emits them as `# bind your real tool` stubs.
- **Studio: full agent IDE** — 13 modules: Build (with **one-click auto-repair** of taint violations, tool quick-picker, per-node provider), Run (**visual execution trace on the canvas** with a scrubber + per-step "why"; Text/URL-fetch/File input + seed memory), Runs, Evals, Prove, Dataset, Monitor (observability charts), Calibrate, Memory, Versions (+ diff), **Audit** (tamper-evident hash-chained log + SIEM export), SDK, plus a guided tour and a plain-English glossary/"why" on every verdict.
- **Export** an agent as a runnable Python script, or a signed **proof certificate** (SHA-256).
- Example agents (`examples/agents/`) + `examples/audit.py` — a reproducible audit finding **5 of 7 common agent patterns have an unguarded prompt-injection path**. Includes the **LangGraph SQL agent** modeled and proven read-only + injection-safe.

### Fixed
- **Boolean obligations** (e.g. `read_only == True`) crashed the Z3 symbolic-SAT path (every variable was typed as a Real). The prover now infers boolean variables. Surfaced by modelling a real OSS agent.

## [0.5.0]

### Added
- **Agent IDE** in the studio: **Build** (click a node, configure the full `Node` -- provider, model, system prompt, extraction schema, obligations, capability, consensus, confidence, transitions) and **Run** (execute the whole agent end-to-end through the real engine, with a live per-step trace + emitted contract); Settings with save/load. Engine: `"END"` is now a universal terminal (a node may end the run whether or not it declares transitions).
- Hooks SDK (`aura_state.hooks`) -- verify your agent's output in your own code and stream it to a running studio: `Monitor` client, `@verified(...)` decorator (fail-closed with `strict=True`), and a `verify()` helper. Works with any framework (CrewAI, LangGraph, plain functions).
- Studio: **Monitor** module (live feed of your real agent's verified outputs via `/api/ingest`), **Import data** module (bulk-verify a CSV/JSON dataset against obligations), and a nav grouped by Design & verify / Runtime / Calibration.

## [0.4.0]

### Added
- **Aura Studio** — a local web platform (`pip install "aura-state[ui]"` → `aura-state ui`) that runs the **real** verifiers on your machine, no cloud, no key. Five modules:
  - **Verify design** — build an agent graph, label capabilities, add Z3 obligations → PROVEN/VIOLATED with counterexamples, a downloadable audit contract, and counterexample-guided **Repair** (violating path glows on the canvas).
  - **Prove data** — Z3 point-check on any data + symbolic obligation-consistency (catches self-contradictory specs).
  - **Live agent** — run a **real** model (local Ollama, or OpenAI/Gemini/DeepSeek with your own key) and prove its output with Z3.
  - **Uncertainty** — conformal prediction intervals, plus **PASC** for end-to-end pipeline coverage.
  - **Risk control** — Conformal Risk Control: calibrate an act/abstain gate with a provable false-action bound.
- **Provider-agnostic engine** — `AuraEngine` accepts any OpenAI-compatible client (Gemini, DeepSeek, Together, local via Ollama/vLLM) or a pre-patched instructor client; the provider layer falls back to the sole client so any model name routes.
- **Cookbook** (`examples/cookbook/`) — realistic agents verified end to end; a "verify an existing LangGraph/CrewAI agent" sidecar; a "same code, any provider" recipe.
- **LangGraph integration** (`examples/integrations/`) — a real LangGraph agent on a local Ollama model, verified by Aura-State.
- **Real-data examples** (`examples/real_data/`) — Z3 verifies 1,000 real public sales records (3,000 obligations); conformal hits 91.3% coverage on 442 real diabetes records.
- **Docs** — honest capability comparison vs LangGraph / CrewAI / Guardrails (`docs/COMPARISON.md`); a logo.
- `aura-state` CLI (`ui`, `version`).

### Notes
- The core library stays dependency-light; the UI is an optional `[ui]` extra (FastAPI + uvicorn).
- 139 tests, all exercising the real solvers/estimators.

## [0.2.1]
- README: pip-friendly quickstart, PyPI badge, corrected test count.

## [0.2.0]
- First public release: verified `process()` loop (Z3 obligations in the extract→verify→retry loop), CTL model checking, static taint (node + field level), conformal risk control, PASC, design→contract compiler, counterexample-guided replanning. Fail-closed proof engine (AST→Z3, no eval), no-exec sandbox, tamper-evident JSON traces.
