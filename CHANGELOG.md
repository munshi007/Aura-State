# Changelog

All notable changes to Aura-State. Format loosely follows Keep a Changelog.

## [0.8.12]

### Added
- **Taint violations show inline in the Design Proof panel.** When "Taint dataflow" reads *violated*, each offending path (`source → sink — no sanitizer`) is listed right under it and is clickable to jump to the sink — the same treatment the lethal-trifecta row already had. (Detail was previously only on the node/graph inspector.)

## [0.8.11]

### Fixed
- **Refreshed the studio's model presets** — the front-end `MODEL_PRESETS` still listed retired Gemini models (`gemini-2.0-flash`, `1.5-*`); now `gemini-flash-latest` / `gemini-3.6-flash` / `gemini-pro-latest` / `gemini-3.6-pro`. The node's Model field is free text (type any id), and switching a node's provider now pulls the default model from the backend (`/api/providers`) so there's one source of truth.

## [0.8.10]

### Fixed
- **The per-node model is honored (customizable again).** The run now uses each node's Model field (editable in the inspector), not a forced provider default. Switching provider updates the LLM nodes' model to that provider's default for you, and you can still change it per node.
- **Refreshed the Gemini default model** `gemini-2.0-flash` → `gemini-3.6-flash` (Google retired the old one). Any model name works — type your own in a node's Model field.

## [0.8.9]

### Added
- **Provider picker next to Run.** Choose the model provider (ollama / openai / gemini / deepseek) right in the top bar — Run uses it immediately, no trip to Settings. Providers without a key show "· no key" and selecting one jumps you to Settings → Providers to add it.

## [0.8.8]

### Fixed
- **Run errors are now visible.** A per-node failure (e.g. an LLM connection error) was recorded in the trace but never surfaced, so a failed Run looked like "nothing happened". It now raises a toast, and the ollama-not-running case gets a clear message ("Can't reach Ollama … pick a provider with a key in Settings").
- **Switching provider actually switches the model.** A node that inherits the agent's provider now also inherits that provider's default model, so selecting `gemini` runs `gemini-2.0-flash` instead of sending a baked-in local model name (`qwen2.5:0.5b`) to the cloud API.

## [0.8.7]

### Fixed
- **Provider key entry is now one step and actually validates.** "Save key" and "Test" both save the key you typed first (so clicking Test before Save no longer reports "not set"), and "Test" makes a real `models.list()` call — it reports `key valid` for a good key or `invalid API key` for a bad one, instead of a vacuous "credentials present".

## [0.8.6]

### Added
- **Enter API keys in the studio.** Settings → Providers now has a key field per provider (OpenAI / Gemini / DeepSeek): paste a key and it's usable immediately — no shell export, no restart. Keys are held **in memory for the session only** and never written to disk (env vars set before launch still work too). New `POST /api/providers/key` endpoint.

## [0.8.5]

Pre-launch hardening: a clean-room install of the published wheel + an adversarial fuzz sweep of the importers and analyzers.

### Fixed — trifecta fail-opens (a verifier must never call a lethal agent safe)
- **Node `kind` no longer disables classification.** The trifecta name-heuristics only ran for `kind == "tool"`, so labelling a `send_email` node `kind: "decision"` made it vanish and the agent verified "safe". Any node that declares a `tool_name` is now classified whatever its `kind`.
- **A mislabeled `sanitizer` can't erase a sink.** A node marked `sanitizer` that also looks like an external sink/source is a misconfiguration; the sanitizer role is dropped (fail-closed, so the sink is caught) and the node is flagged for review.
- **Ambiguity fails closed.** When unclassifiable tools coexist with a reachable untrusted source *and* a private read, one of them could be the exfil leg — `check` now reports this as blocking ("cannot prove trifecta-free") instead of a green pass. An import that recognized no tools is no longer reported as PROVEN.

### Fixed — robustness & security
- The importers and `check_flow` return clean errors instead of tracebacks on malformed input (non-dict annotations, unhashable/missing tool names, non-UTF8 files, nodes without a string `id`, bad edges).
- **SSRF guard** on the studio's `/api/fetch_url`: private / loopback / link-local / cloud-metadata targets (127.0.0.1, 169.254.169.254, 10.x, …) are refused.
- README gains a **"Scope & honest limits"** section and corrects the conformal description (repeated same-input runs are dispersion, not calibrated coverage).

Execution-safety is unchanged and re-confirmed: importing an agent's code/obligations never runs them (AST-only).

## [0.8.4]

### Fixed
- **Trifecta false positives from prompt prose.** Role classification blended a node's free-text `system_prompt` into the name heuristic, so a payment sink whose prompt read *"Issue the refund to the customer account"* was wrongly tagged untrusted+private and reported as a self-referential lethal trifecta. Classification now uses the tool name + an explicit tool description only. (Found by driving the studio.)
- **Studio imports no longer force every tool to a sink.** An imported tool with unknown/read/local-write side-effect stays `plain` (only a declared external tool is a sink), so the classifier decides its role by name instead of flagging all imported tools as exfil.
- The studio status bar reads the version live from a new `/api/version` endpoint instead of a hardcoded string.

## [0.8.3]

### Added
- **Studio: regression gate in the Audit module.** Save the current design as a baseline when it's clean; after edits, *Re-check* diffs against it and flags only the **new** blocking findings (with a "✓ no regressions" / "✕ N new blocking" verdict and a resolved count) — the CI `aura-state check --baseline` gate, now interactive. Backed by a new `/api/check` endpoint that returns the full analyzer's findings with their stable baseline keys. Baselines persist per agent in the browser.

## [0.8.2]

### Added
- **Studio: "Import from code"** — the source-code importer is now in the UI (agent menu → *Import from code…*): paste a LangGraph/CrewAI/LangChain agent and it builds the flow and verifies the trifecta, via a new `/api/code/import` endpoint (parses with `ast`, never runs the code). The studio now surfaces all three import paths (MCP tools, agent source, JSON).
- **Resizable & collapsible panels** — drag the divider on either side to resize the Nodes tree and the Inspector; a ‹/› button collapses each (with a slim reopen tab), persisted per browser. The status bar now shows the correct version.

## [0.8.1]

### Fixed
- **Enum constraints are now enforced.** `schema_compiler` compiles a JSON-Schema `enum` to `typing.Literal`, so an out-of-enum value fails validation (previously the enum was only stored in `json_schema_extra` and never checked).
- **Honest telemetry** (tasks 0004/0007): the per-node "conformal" report over repeated consensus runs is relabelled `consensus_dispersion` — re-running the same input measures agreement/dispersion, not calibrated predictive coverage. Provider cost tracking now reads real token usage off the fallback `create()` path (`_raw_response`) instead of silently recording 0 tokens, and warns once on an unpriced model instead of a silent $0.
- Deduped `analyze_taint` (the same source→sink reached by multiple paths was reported repeatedly); removed an unused import.

_Not changed (intentional):_ a decision node whose `sandbox_rule` evaluates falsy is a valid routing branch, not a provable-obligation failure, so it is not treated as a hard verification error (its Z3 obligations still fail closed).

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
