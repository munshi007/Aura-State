# Changelog

All notable changes to Aura-State. Format loosely follows Keep a Changelog.

## [0.8.0]

### Added
- **Source-code importer** (`aura_state.loaders.code`) — `aura-state check` now accepts a **LangGraph / CrewAI / LangChain agent's source** (a `.py` file or a whole directory) and statically extracts its tool surface: `@tool`/`@tool("name")` functions, `Tool(...)`/`StructuredTool.from_function(...)`, and framework tool classes (`SerperDevTool`, `FileReadTool`, `CodeInterpreterTool`, …) mapped to known roles. It builds the same worst-case hub flow as the MCP importer and runs the trifecta/taint checks. Parses with `ast` and **never imports or executes the code** — reading an agent's repo must not run it. Examples in `examples/code/`.
- `_load_flow` for `.py` no longer executes the module: an explicit `flow`/`FLOW` dict is read with `ast.literal_eval` (safe), otherwise the tool surface is imported statically. Directories are supported.

## [0.7.2]

A full end-to-end code review (5 parallel reviewers) surfaced real defects; all fixed with regression tests. 199 tests.

### Fixed — trifecta false-negatives (the flagship must never fail open)
- **Un-annotated MCP tools no longer verify a lethal surface as safe.** The importer defaulted tools with no annotation hints to `write`, which the classifier skipped for the untrusted/private legs — so `fetch + read_file + create_issue` (no hints) reported *safe*. `_side_effect` now returns `None` (unknown reach) and any tool that can't be placed in a role is surfaced as an advisory, not silently passed.
- **Private/untrusted classification is no longer gated by `side_effect`.** A single tool that both reads private data and sends externally (e.g. `email_customer_record`) was tagged exfil-only and missed; it now closes the trifecta correctly.
- **Compound tool names are matched.** Exfil detection tokenizes camelCase/snake_case, so `post_update` / `slack_post_message` / `postMessage` register as exfil while `postgres` does not.
- A single node that is both untrusted-source and external sink is now flagged.

### Fixed — verifier correctness
- **`proof_engine` division:** `/` now compiles to true (Real) division and `//` to floor, instead of both truncating as Z3 Int division (which marked `avg == 7/2 == 3` satisfied). Z3 `unknown` now fails closed.
- **Engine fails closed:** an extraction that never satisfies its obligations is no longer acted on — `process()` raises `MaxRetriesExceededError` unless a risk-controlled escalation is configured. Health/router metrics now record the real outcome instead of a hardcoded success.
- **`pipeline_conformal`** `covers()`/`interval()` now raise when uncalibrated instead of vacuously covering everything; **`conformal`** exposes the honest jackknife+ `worst_case_coverage` (1−2α).
- **CTL reachability/completion** in `check` now use the declared `entry` node.

### Fixed — the CI regression gate
- `check --json --baseline` now applies the regression gate (was ignored, so JSON/CI mode failed on pre-existing debt).
- Baseline finding identity now includes a source→sink discriminator, so a **new** untrusted source reaching an **existing** sink is correctly flagged as a regression instead of tagged `[known]`.

### Fixed — other
- Sandbox `**` is bounded (a single `9**9**9**9` could hang the process); MCP importer no longer drops same-named tools from different servers; `json_graph` routing no longer silently falls through to the first edge; `schema_compiler` no longer silently remaps unknown fields to a fuzzy-nearest name; studio taint/edit UI marks the correct sink and clears stale verdicts; dead code removed.

## [0.7.1]

### Added
- **Real-world trifecta audit** (`examples/audit/` + `docs/AUDIT.md`) — reproducible lethal-trifecta analysis of real MCP-server compositions (GitHub, filesystem, fetch, Slack, Postgres, Brave) and agent frameworks (CrewAI, AutoGPT), modeled faithfully from their published tool surfaces with sources. **5 of 6 can close the trifecta**, including the exact shape of the Invariant Labs GitHub-MCP exploit. `python examples/audit/run.py` prints the table; each target is one `uvx aura-state check` away.

### Fixed
- **Trifecta role classifier — fewer false positives.** A read-only tool is no longer treated as an external-comms sink just because its name matches an exfil verb (e.g. `slack_list_channels`), and the MCP importer now uses `openWorldHint` to distinguish external comms (`slack_post_message`, `add_issue_comment`) from local mutations (`write_file`), which are dangerous sinks but not the trifecta's exfil leg.

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
