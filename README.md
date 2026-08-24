# Aura-State

A Python framework for building LLM workflows as state machines, with formal verification built in.

```bash
pip install git+https://github.com/munshi007/Aura-State.git
```

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

### Confidence intervals on extractions

Run the extraction multiple times and get distribution-free confidence intervals:

```python
from aura_state import conformal_interval

budgets = [450000, 452000, 448000, 450000, 451000]
ci = conformal_interval(budgets, confidence=0.95)
# ci.lower = 447800, ci.upper = 452200
```

This uses conformal prediction (Vovk et al., 2005) — no distributional assumptions required.

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

## Benchmark results

We ran 10 real-estate sales transcripts through a 4-node pipeline using GPT-4o-mini (30 API calls total):

```
Field             Accuracy
──────────────   ──────────
name                  100%
budget                100%
bedrooms              100%
pre_approved           90%
timeline               90%
city                   80%

Temporal properties:       3/3 proven
Z3 proof obligations:     20/20 passed
Avg latency:              1.4s
```

```bash
# See verification reject a hallucination in the loop — no API key needed
python examples/verified_loop_demo.py

# Full pipeline benchmark — no API key needed
python examples/benchmark/run_benchmark.py

# With real LLM calls (needs OPENAI_API_KEY in .env)
python examples/benchmark/run_live.py --model gpt-4o-mini --runs 3
```

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
pip install git+https://github.com/munshi007/Aura-State.git
```

Python 3.10+ required. Dependencies: `pydantic`, `instructor`, `openai`, `networkx`, `pyModelChecking`, `z3-solver`, `pyyaml`.

## Tests

```bash
python -m pytest tests/ -v
# 100 tests passing
```

## Docs

- [Usage Guide](docs/GUIDE.md) — code examples for every feature
- [Algorithm Reference](docs/ALGORITHMS.md) — deep-dive into CTL, Z3, Thompson sampling, conformal prediction
- [Contributing](CONTRIBUTING.md) — architecture overview and how to contribute
- [Benchmark](examples/benchmark/) — synthetic and live benchmarks

## License

MIT
