<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo.svg" alt="Aura-State" width="104" height="104">
  </picture>
</p>

<h1 align="center">Aura-State</h1>

<p align="center"><b>Build LLM agents you can actually prove things about.</b></p>

<p align="center">
  Verification that runs <i>in the loop</i>, not the sidebar — Z3 proofs, CTL model checking, and conformal risk control gate every step. A value that can't be proven is never accepted.
</p>

<p align="center">
  <a href="https://pypi.org/project/aura-state/"><img alt="PyPI" src="https://img.shields.io/pypi/v/aura-state.svg?color=3d3aa8"></a>
  <img alt="CI" src="https://github.com/munshi007/Aura-State/actions/workflows/ci.yml/badge.svg">
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-3d3aa8.svg">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="tests" src="https://img.shields.io/badge/tests-136%20passing-1c8a5b.svg">
</p>

```bash
pip install aura-state
```

## See it in 10 seconds (no API key)

Paste this after installing — Z3 rejects a hallucinated value and accepts a correct one, in the loop:

```python
from aura_state import prove_extraction

print(prove_extraction({"area": 100, "rate": 3, "total": 999}, ["total == area * rate"]).verified)  # False
print(prove_extraction({"area": 100, "rate": 3, "total": 300}, ["total == area * rate"]).verified)  # True
```

For the full runnable proofs against the real solvers — each ~10s, no API key — clone the repo:

```bash
git clone https://github.com/munshi007/Aura-State && cd Aura-State
pip install -e .
```

| Demo | What it proves |
|---|---|
| `python examples/verified_loop_demo.py` | Z3 rejects a hallucinated extraction in the loop, retries, accepts |
| `python examples/taint_proof_demo.py` | untrusted input provably can't reach a dangerous tool |
| `python examples/risk_abstention_demo.py` | acts only within a calibrated risk budget, else escalates to a human |
| `python examples/emit_contract_demo.py` | a portable contract compiled faithfully from the design |
| `python examples/replan_demo.py` | the verifier *repairs* the plan until it's proven-safe |
| `python examples/pasc_demo.py` | pipeline-aware conformal calibrates the end-to-end answer, not just each step |

## Local studio — click, don't code (no cloud, no key)

Prefer a UI? Launch a web app that runs the **real** verifiers on your own machine:

```bash
pip install "aura-state[ui]"
aura-state ui          # opens http://127.0.0.1:8155 in your browser
```

Build an agent graph visually, label capabilities (untrusted / sink / sanitizer),
add Z3 obligations, and hit **Verify** — the local backend runs the actual Z3
proofs, CTL model checking, and static taint analysis, shows the verdicts and
counterexamples, and lets you download the audit contract. Nothing leaves your
machine — no cloud, no API key.

## What this is

Most LLM frameworks let you chain API calls and hope for the best. Aura-State takes a different approach: you define your workflow as a typed graph of nodes, and **verification runs inside the loop** — every extraction must satisfy its formal contract before the workflow moves on.

The key difference is what happens between nodes:

- **Extractions** are checked against Z3 proof obligations, in the extract→verify→retry loop — a value that can't be proven is not accepted (fail-closed)
- **Math** runs in a no-`exec` sandboxed interpreter, never hallucinated
- **Uncertainty** is a real conformal interval over repeated runs, not a vibe
- **Workflows** are model-checked (CTL) for reachability/completion/ordering *before* they run
- **Routing** (when a node returns an ambiguous edge) is a Thompson-sampling bandit, not an LLM guess

## Quick example

```python
from aura_state import AuraEngine, Node, CompiledTransition
from pydantic import BaseModel, Field
from openai import OpenAI

# Define what you want to extract
class LeadData(BaseModel):
    name: str = Field(description="Full name")
    budget: int = Field(description="Budget in USD")
    timeline: str = Field(description="Buying timeline")

# Define a node that extracts it — with a Z3 obligation the value must satisfy
class ExtractLead(Node):
    system_prompt = "Extract lead info from a sales call transcript."
    extracts = LeadData
    obligations = ["budget > 0"]   # proven in the loop; unprovable -> not accepted

    def handle(self, user_text, extracted_data=None, memory=None):
        return "QualifyBudget", extracted_data.model_dump()

# Define a decision node that does deterministic math (no LLM). Its rule runs
# even though the node does no extraction — it reads prior state from memory.
class QualifyBudget(Node):
    system_prompt = "Score the lead."
    sandbox_rule = "result = budget > 100000"  # runs in the no-exec sandbox

    def handle(self, user_text, extracted_data=None, memory=None):
        return "END", memory

# Wire it up
engine = AuraEngine(llm_client=OpenAI())
engine.register(ExtractLead, QualifyBudget)
engine.connect([
    CompiledTransition(from_node=ExtractLead, to_node=QualifyBudget),
])

# Run
next_state, data = engine.process("ExtractLead", user_text="Hi, I'm Sarah. Budget is $450k.")
```

## What happens under the hood

When you call `engine.process()`, it runs through these steps in order:

```
1. Few-shot injection      →  optional: inject similar past successes as examples.
2. Verification loop       →  extract → check (sandbox rule + Z3 obligations) → retry.
                              A value that fails its contract is not accepted (fail-closed).
3. Conformal interval      →  with consensus > 1, build a real interval over the runs.
4. Your handle() method    →  your routing / business logic runs here.
5. Bandit router           →  if handle() returns an invalid edge, Thompson-sample a feasible one.
6. State serialization     →  save state (JSON, tamper-evident) for time-travel debugging.
```

Graph-level properties (reachability, completion, ordering) are checked separately
at **design time** with `engine.verify([...])` — CTL model checking over the whole
graph, which per-transition checks can't do. Everything above happens *in the loop*;
`engine.verification_reports()` returns the proof results and intervals per step.

## Formal verification (the interesting part)

This is what actually makes Aura-State different from other frameworks.

### Verify your workflow graph before it runs

Your node graph gets compiled into a [Kripke structure](https://en.wikipedia.org/wiki/Kripke_structure) and checked against temporal logic properties:

```python
from aura_state import verify_engine, reachability, mutual_exclusion, eventual_completion

results = verify_engine(engine, [
    {"description": "QualifyBudget is reachable", "formula": reachability("QualifyBudget")},
    {"description": "All paths terminate", "formula": eventual_completion("QualifyBudget")},
])
# Result: PROVEN or VIOLATED, with the exact states that satisfy/violate
```

This is the same technique used to verify hardware circuits and flight control systems (CTL model checking, Clarke et al. 1986).

### Prove that extracted data is correct

After the LLM extracts values, Z3 (a theorem prover from Microsoft Research) can formally prove they satisfy your constraints:

```python
from aura_state import prove_extraction

result = prove_extraction(
    {"budget": 450000, "cost_per_sqft": 3, "total": 1350000},
    obligations=["budget > 0", "total == budget * cost_per_sqft"],
)
# result.verified = True
# If False, Z3 gives you a counterexample showing exactly what broke
```

It also proves your obligations aren't *self-contradictory*. `["x > 5", "x < 3"]`
can never hold — Z3 catches that symbolically (variables ranging freely over the
declared field bounds, not pinned to one value), and the design→contract compiler
flags it per node before you ship.

### Confidence intervals on extractions

Run the extraction multiple times and get distribution-free confidence intervals:

```python
from aura_state import conformal_interval

budgets = [450000, 452000, 448000, 450000, 451000]
ci = conformal_interval(budgets, confidence=0.95)
# ci.lower = 447800, ci.upper = 452200
```

This uses conformal prediction (Vovk et al., 2005) — no distributional assumptions required.

**Pipeline-aware (PASC):** a 95% guarantee at each node is *not* 95% end-to-end —
errors compound. `PipelineConformal` calibrates on the composed output so the
guarantee holds for the final answer. In the demo, per-step conformal covers the
end-to-end result only ~48%; PASC hits the nominal 90%. See
`python examples/pasc_demo.py`.

### Compile a runtime contract from the design

The obligations, CTL verdicts, and confidence a workflow was proven against
compile into a single portable, versioned contract. Because it's derived from
the same typed design the engine runs, the specification is **faithful by
construction** — spec and implementation are one artifact and can't drift.

```python
contract = engine.compile_contract(properties=[
    {"description": "RouteLead is reachable", "formula": reachability("RouteLead")},
])
contract.to_json()                        # portable, content-addressable
check_faithfulness(contract, "QualifyLead", extracted)   # contract agrees with the loop
diff_contracts(old, contract)             # design-time regression gate
```

Every other assurance system *consumes* a behavioral contract it can't author —
and hand-written policy drifts from the code (and is only 24–35% faithful when
translated from prose). Here the contract is emitted from the design that was
proven. See `python examples/emit_contract_demo.py`.

### Prove untrusted data can't reach a dangerous tool (injection-proof)

Label nodes with capability types and the compiler statically proves — over the
typed graph — that no untrusted source can reach a dangerous sink without
passing a sanitizer. It tracks *provenance, not content*, so it can't be fooled
by the encodings that defeat runtime scanners. The verdict compiles into the
contract, so a runtime can refuse to deploy a `VIOLATED` graph.

```python
class Ingest(Node):    untrusted_source = True     # LLM / external tool output
class Review(Node):    sanitizer = True            # clears taint
class SendEmail(Node): dangerous_sink = True       # irreversible action

analyze_taint(engine)   # -> VIOLATED (Ingest -> SendEmail) unless Review is in the path
```

It's **field-level**: label individual fields, and a clean field passes a sink
untouched while only a *tainted* field reaching it is a violation — with the exact
field and its origin named. A field-specific sanitizer clears just its field.

```python
class Ingest(Node): untrusted_fields = ["note"]   # free text is untrusted
class Send(Node):   sink_fields = ["account_id"]  # the action consumes account_id

analyze_field_taint(engine)   # PROVEN — the tainted `note` never reaches the sink arg
```

Everyone else sells injection *detection* (probabilistic). This is
*impossibility* over the design. See `python examples/taint_proof_demo.py`.

### Act only if calibrated risk ≤ ε, otherwise escalate

The "knows when it doesn't know" story made into an actual gate. Calibrate a
controller on a labeled set and the agent auto-acts only when its false-action
rate is provably within budget — everything below the threshold escalates to a
human, never a silent guess.

```python
ctrl = RiskController(epsilon=0.05).calibrate(scores, correct)   # false-action rate ≤ 5%

class Decide(Node):
    risk_controller = ctrl
    escalation_node = "HumanReview"
    def risk_score(self, extracted_data=None, conformal=None, memory=None):
        return confidence   # in [0,1]
```

Uses Conformal Risk Control (arXiv:2208.02814); Learn-Then-Test
(arXiv:2110.01052) for tuning several thresholds. Abstention is a first-class
engine outcome. See `python examples/risk_abstention_demo.py`.

### Let the verifier repair the plan (counterexample-guided replanning)

Verification is usually a gate that says VIOLATED and stops. Here the
counterexample — a tainted path, a CTL violating state, a Z3 assignment — is fed
back to a replanner, which edits the plan and re-verifies, until it's proven or a
budget is hit. The plan is provably correct *because* the verifier drove it there.

```python
result = engine.repair()          # verify → counterexample → repair → re-verify
result.verified                   # True: driven to PROVEN (e.g. a sanitizer inserted)
result.unresolved                 # if it aborts: the explicit remaining violations
```

Never a silent pass — an unrepairable design aborts with the violation named.
Refs: PAT-Agent (arXiv:2509.23675), VERIMAP (arXiv:2510.17109). See
`python examples/replan_demo.py`.

## Results — real data, reproducible, no API key

Every number below comes from running Aura-State on real, public data (or a real
local model). Nothing here is simulated.

| What | Real subject | Result | Reproduce |
|---|---|---|---|
| **Z3 verifies invariants at scale** | 1,000 real public sales records (3 arithmetic invariants each) | **1,000/1,000 verified**, 3,000 obligations, **~1,200 records/sec**; rejects a corrupted record with the exact failing obligation | `python examples/real_data/verify_real_dataset.py` |
| **Conformal coverage on real data** | 442 real diabetes patient records (scikit-learn) | requested 90% → **91.3% empirical coverage** on held-out patients | `python examples/real_data/conformal_on_real_data.py` |
| **A real LangGraph agent, local model** | `qwen2.5:0.5b` via Ollama — no key, no cloud | agent's extraction **Z3-verified**, tool graph **proven injection-safe**, audit contract emitted | `python examples/integrations/langgraph_verified.py` |

```bash
# real data through the verifier — no LLM, no key
python examples/real_data/verify_real_dataset.py
pip install scikit-learn && python examples/real_data/conformal_on_real_data.py

# a real LangGraph agent on a local model — no key, no cloud
ollama pull qwen2.5:0.5b && pip install langgraph
OLLAMA_MODEL=qwen2.5:0.5b python examples/integrations/langgraph_verified.py

# verification rejects a hallucination in the loop — no key
python examples/verified_loop_demo.py
```

See [`examples/real_data/`](examples/real_data/) and
[`examples/integrations/`](examples/integrations/) for the full runs.

## Project structure

```
aura_state/
├── core/
│   ├── engine.py              # Main engine — verified process() loop + bandit router
│   ├── adaptive_graph.py      # Node health metrics + per-edge Beta-Bernoulli posteriors
│   ├── verification_loop.py   # Extract → verify (sandbox + Z3) → retry loop
│   └── providers.py           # Multi-model routing + cost tracking
├── verification/             # ← the core: correct, adversarially-tested primitives
│   ├── proof_engine.py        # Z3 proofs (fail-closed AST→Z3 compiler, no eval)
│   ├── conformal.py           # jackknife+ prediction intervals (order statistic)
│   └── temporal_verifier.py   # Kripke + CTL model checking (init-state, structural deadlocks)
├── execution/
│   ├── sandbox.py             # No-exec allowlist AST evaluator (deny-by-default)
│   └── tracer.py              # State serialization, tamper-evident JSON (time-travel debug)
├── compiler/
│   ├── schema_compiler.py     # JSON Schema → Node classes
│   └── dspy_tuner.py          # KNN few-shot selection (real embedder required)
├── memory/
│   └── pruner.py              # Context window optimization
└── consensus/
    └── auto_vote.py           # Multi-run extraction with voting
```

## Installation

```bash
pip install aura-state
```

Or the latest from source:

```bash
pip install git+https://github.com/munshi007/Aura-State.git
```

Python 3.10+ required. Dependencies: `pydantic`, `instructor`, `openai`, `networkx`, `pyModelChecking`, `z3-solver`, `pyyaml`.

## Tests

```bash
python -m pytest tests/ -v
# 136 tests passing
```

## Works with any LLM provider

Aura-State's verification is independent of the model — only extraction calls
one, and nearly every provider speaks the OpenAI-compatible API. Point the client
anywhere; the Z3/CTL/taint/conformal code is identical:

```python
from openai import OpenAI
# Gemini, DeepSeek, Together, or a local model via Ollama/vLLM — just the base_url
engine = AuraEngine(llm_client=OpenAI(api_key=key, base_url="https://api.deepseek.com"))
```

See [`examples/cookbook/`](examples/cookbook/) for OpenAI / Gemini / DeepSeek /
local recipes. Your API key is read from your environment — never hard-coded.

## Docs

- [Cookbook](examples/cookbook/) — realistic agents, verified end to end; any provider
- [LangGraph integration](examples/integrations/) — a real LangGraph agent on a local model (Ollama), verified by Aura-State — no API key
- [Real-data verification](examples/real_data/) — Z3 verifies 1,000 real sales records (3,000 obligations, ~1,200/sec); conformal hits 91.3% coverage on real diabetes records — no LLM, no key
- [Comparison](docs/COMPARISON.md) — Aura-State vs LangGraph / CrewAI / Guardrails (they orchestrate; we verify)
- [Usage Guide](docs/GUIDE.md) — code examples for every feature
- [Algorithm Reference](docs/ALGORITHMS.md) — deep-dive into CTL, Z3, Thompson sampling, conformal prediction
- [Contributing](CONTRIBUTING.md) — architecture overview and how to contribute
- [Pipeline walkthrough](examples/benchmark/) — an illustrative end-to-end pipeline (mocked LLM); for measured results see [real-data](examples/real_data/)

## License

MIT
