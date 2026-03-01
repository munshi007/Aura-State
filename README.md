# Aura-State

A Python framework for building LLM workflows as state machines, with formal verification built in.

```bash
pip install git+https://github.com/munshi007/Aura-State.git
```

## What this is

Most LLM frameworks let you chain API calls and hope for the best. Aura-State takes a different approach: you define your workflow as a graph of nodes, each with a specific job, and the framework handles extraction, verification, and routing.

The key difference is what happens between nodes:

- **Routing** is scored mathematically (MCTS), not decided by the LLM
- **Math** runs in a sandboxed interpreter, never hallucinated
- **Extractions** can be formally proven correct using Z3
- **Workflows** can be verified for safety properties before they run

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

# Define a node that extracts it
class ExtractLead(Node):
    system_prompt = "Extract lead info from a sales call transcript."
    extracts = LeadData

    def handle(self, user_text, extracted_data=None, memory=None):
        return "QualifyBudget", extracted_data.model_dump()

# Define a node that does deterministic math (no LLM)
class QualifyBudget(Node):
    system_prompt = "Score the lead."
    sandbox_rule = "result = budget > 100000"  # runs in sandboxed AST, not LLM

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
1. Adaptive DAG health check     →  Should this node be skipped or retried?
2. GraphRAG cache lookup          →  Have we seen this exact input before? Skip the LLM.
3. Few-shot injection             →  Find similar past successes, inject as examples.
4. LLM extraction + verification  →  Extract data, verify with Z3, retry if wrong.
5. Your node's handle() method    →  Your business logic runs here.
6. MCTS Routing (UCB1)        →  Score branches using UCB1 + AdaptiveDAG metrics.
7. State serialization            →  Save state for time-travel debugging.
8. Speculative execution          →  Pre-compute likely next nodes in parallel.
```

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
Routing accuracy:          90%
Avg latency:              1.4s
```

```bash
# Try it yourself — no API key needed
python examples/benchmark/run_benchmark.py

# With real LLM calls (needs OPENAI_API_KEY in .env)
python examples/benchmark/run_live.py --model gpt-4o-mini --runs 3
```

## Project structure

```
aura_state/
├── core/
│   ├── engine.py              # Main engine — process() + MCTS/UCB1 routing
│   ├── adaptive_graph.py      # Node health monitoring
│   ├── verification_loop.py   # Extract → verify → retry loop
│   └── providers.py           # Multi-model routing + cost tracking
├── compiler/
│   ├── schema_compiler.py     # JSON Schema → Node classes
│   └── dspy_tuner.py          # KNN few-shot selection
├── verification/
│   ├── temporal_verifier.py   # Kripke + CTL model checking
│   ├── conformal.py           # Conformal prediction intervals
│   └── proof_engine.py        # Z3 proofs
├── execution/
│   ├── tracer.py              # State serialization (time-travel debug)
│   └── sandbox.py             # Safe math execution (AST validated)
├── memory/
│   ├── trajectory_cache.py    # Subgraph isomorphism cache
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
# 65 tests passing
```

## Docs

- [Usage Guide](docs/GUIDE.md) — code examples for every feature
- [Algorithm Reference](docs/ALGORITHMS.md) — deep-dive into CTL, Z3, MCTS, UCB1, conformal prediction
- [Contributing](CONTRIBUTING.md) — architecture overview and how to contribute
- [Benchmark](examples/benchmark/) — synthetic and live benchmarks

## License

MIT
