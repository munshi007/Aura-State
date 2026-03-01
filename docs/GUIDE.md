# Aura-State Usage Guide

A complete guide to building, verifying, and running LLM workflows with Aura-State.

## Table of Contents

- [Core Concepts](#core-concepts)
- [Defining Nodes](#defining-nodes)
- [Structured Extraction](#structured-extraction)
- [Wiring a Pipeline](#wiring-a-pipeline)
- [Running the Pipeline](#running-the-pipeline)
- [Declarative JSON Workflows](#declarative-json-workflows)
- [Formal Verification](#formal-verification)
- [Z3 Proof Engine](#z3-proof-engine)
- [Conformal Prediction](#conformal-prediction)
- [Sandbox Rules](#sandbox-rules)
- [Multi-Provider Routing](#multi-provider-routing)
- [Cost Tracking](#cost-tracking)
- [Consensus Extraction](#consensus-extraction)
- [Time-Travel Debugging](#time-travel-debugging)

---

## Core Concepts

Aura-State models every LLM workflow as a **directed graph of Nodes**. Each Node has:

- A **system prompt** that tells the LLM what to do
- An optional **Pydantic schema** for structured extraction
- A **handle()** method where you write your business logic
- Optional **sandbox rules** for deterministic math

The engine handles everything else: LLM calls, retries, caching, routing, verification.

```
User Input → Node A (extract) → Node B (calculate) → Node C (decide) → Output
```

You define the nodes. The engine compiles them into a state machine and runs them.

---

## Defining Nodes

A Node is a Python class that inherits from `Node`:

```python
from aura_state import Node

class GreetUser(Node):
    system_prompt = "Greet the user by name."

    def handle(self, user_text, extracted_data=None, memory=None):
        return "NextNode", {"message": f"Hello! You said: {user_text}"}
```

The `handle()` method returns two things:
1. The **name of the next node** (or `"END"` to stop)
2. A **payload dict** that gets passed forward as memory

---

## Structured Extraction

Use Pydantic models to extract structured data from unstructured text:

```python
from pydantic import BaseModel, Field
from aura_state import Node

class InvoiceData(BaseModel):
    vendor: str = Field(description="Name of the vendor")
    amount: float = Field(description="Total amount in USD")
    due_date: str = Field(description="Payment due date")

class ExtractInvoice(Node):
    system_prompt = "Extract invoice details from the document text."
    extracts = InvoiceData

    def handle(self, user_text, extracted_data=None, memory=None):
        # extracted_data is a populated InvoiceData instance
        invoice = extracted_data.model_dump()
        return "ValidateInvoice", invoice
```

When `extracts` is set, the engine uses [Instructor](https://github.com/jxnl/instructor) to force the LLM to return data matching your schema. No parsing, no regex, no hoping.

---

## Wiring a Pipeline

Connect nodes with `CompiledTransition`:

```python
from aura_state import AuraEngine, CompiledTransition
from openai import OpenAI

engine = AuraEngine(llm_client=OpenAI())

engine.register(ExtractInvoice, ValidateInvoice, ApprovePayment)

engine.connect([
    CompiledTransition(from_node=ExtractInvoice, to_node=ValidateInvoice),
    CompiledTransition(from_node=ValidateInvoice, to_node=ApprovePayment),
])
```

You can have **branching** — one node can connect to multiple targets:

```python
engine.connect([
    CompiledTransition(from_node=ReviewNode, to_node=ApproveNode),
    CompiledTransition(from_node=ReviewNode, to_node=RejectNode),
])
```

When a node returns a target that has multiple options, the engine uses MCTS (Monte Carlo Tree Search) to score and select the best path.

---

## Running the Pipeline

```python
next_state, data = engine.process(
    current_state="ExtractInvoice",
    user_text="Invoice from Acme Corp. Total: $12,500. Due: March 15, 2026.",
)

print(next_state)  # "ValidateInvoice"
print(data)        # {"vendor": "Acme Corp", "amount": 12500.0, "due_date": "March 15, 2026"}
```

Every call to `engine.process()` runs through the full pipeline:
1. Adaptive DAG health check
2. GraphRAG cache lookup (skip LLM if seen before)
3. Few-shot teleprompting (inject past successes)
4. Verification loop (extract → verify → reflect → retry)
5. Your node's `handle()` logic
6. MCTS routing (if multiple targets)
7. AuraTrace state serialization
8. Speculative pre-computation of next nodes

---

## Declarative JSON Workflows

You can define workflows in JSON instead of Python:

```json
{
  "nodes": [
    {
      "id": "ExtractDimensions",
      "system_prompt": "Extract room measurements from the input.",
      "extracts": {
        "wall_area": {"type": "float", "description": "Wall area in sqft"},
        "ceiling_height": {"type": "float", "description": "Ceiling height in feet"}
      }
    },
    {
      "id": "CalculateCost",
      "system_prompt": "Calculate the total cost."
    }
  ],
  "edges": [
    {"from": "ExtractDimensions", "to": "CalculateCost"}
  ]
}
```

Load it:

```python
from aura_state import AuraEngine, JSONGraphLoader
from openai import OpenAI

engine = AuraEngine(llm_client=OpenAI())
JSONGraphLoader.load("flow.json", engine)

next_state, data = engine.process("ExtractDimensions", user_text="850 sqft room, 9ft ceilings")
```

---

## Formal Verification

Verify your workflow is safe **before** it ever runs. Aura-State compiles your node graph into a [Kripke structure](https://en.wikipedia.org/wiki/Kripke_structure) and checks [CTL properties](https://en.wikipedia.org/wiki/Computation_tree_logic) using model checking.

```python
from aura_state import verify_engine, reachability, mutual_exclusion, eventual_completion

results = verify_engine(engine, [
    {
        "description": "Payment approval is reachable",
        "formula": reachability("ApprovePayment"),
    },
    {
        "description": "Cannot approve and reject simultaneously",
        "formula": mutual_exclusion("ApprovePayment", "RejectPayment"),
    },
    {
        "description": "Every path eventually terminates",
        "formula": eventual_completion("ApprovePayment", "RejectPayment"),
    },
])

for r in results:
    print(f"{r.result.value}: {r.property_text}")
    # "proven: Payment approval is reachable"
    # "proven: Cannot approve and reject simultaneously"
    # "proven: Every path eventually terminates"
```

Available property constructors:

| Function | CTL Formula | Meaning |
|----------|------------|---------|
| `reachability("X")` | EF(X) | X is reachable from at least one path |
| `mutual_exclusion("X", "Y")` | AG(¬(X ∧ Y)) | X and Y can never be active simultaneously |
| `eventual_completion("X", "Y")` | AF(X ∨ Y) | Every path eventually reaches X or Y |
| `always_before("X", "Y")` | AG(Y → X) | Y is only reachable if X was visited |

---

## Z3 Proof Engine

Formally prove that LLM-extracted data satisfies your business rules using the [Z3 theorem prover](https://github.com/Z3Prover/z3):

```python
from aura_state import prove_extraction

data = {"area": 500, "cost_per_sqft": 3, "total_cost": 1500}

result = prove_extraction(data, [
    "area > 0",
    "cost_per_sqft > 0",
    "total_cost == area * cost_per_sqft",
])

print(result.verified)           # True
print(result.failed_obligations) # []
```

When a proof fails, Z3 generates a counterexample:

```python
bad_data = {"area": 500, "cost_per_sqft": 3, "total_cost": 9999}

result = prove_extraction(bad_data, ["total_cost == area * cost_per_sqft"])

print(result.verified)           # False
print(result.failed_obligations) # ["total_cost == area * cost_per_sqft"]
print(result.counterexample)     # {"extracted_values": {...}, "failed_constraints": [...]}
```

Use `prove_consistency` for cross-field relationships:

```python
from aura_state import prove_consistency

result = prove_consistency(
    {"unit_cost": 10, "quantity": 5, "total": 50},
    relationships=["total == unit_cost * quantity"],
)
```

---

## Conformal Prediction

Get statistically guaranteed confidence intervals on extracted values. Run the extraction multiple times (using consensus), then compute prediction intervals:

```python
from aura_state import conformal_interval, conformal_from_extractions

# From multiple extraction runs on the same input
budgets = [450000, 452000, 448000, 450000, 451000]

ci = conformal_interval(budgets, confidence=0.95)

print(ci.point_estimate)  # 450000
print(ci.lower)          # 447800
print(ci.upper)          # 452200
print(ci.confidence)     # 0.95
```

Or from Pydantic model instances:

```python
extractions = [model1, model2, model3, model4, model5]
result = conformal_from_extractions(extractions, confidence=0.95)

for field, interval in result.intervals.items():
    print(f"{field}: [{interval.lower}, {interval.upper}]")
```

---

## Sandbox Rules

For math that must be deterministic (never hallucinated), use sandbox rules:

```python
class CalculateCost(Node):
    system_prompt = "Calculate the project cost."
    sandbox_rule = "result = wall_area * cost_per_sqft * 1.15"

    def handle(self, user_text, extracted_data=None, memory=None):
        return "END", memory
```

The sandbox compiles the rule into a Python AST, validates it for safety (no imports, no I/O, no exec), and runs it deterministically. The LLM never touches this calculation.

---

## Multi-Provider Routing

Route different nodes to different LLM providers:

```python
engine = AuraEngine(llm_client=OpenAI())

# Register additional providers
engine.provider.register_client("gpt", instructor.from_openai(OpenAI()))
engine.provider.register_client("anthropic", instructor.from_anthropic(Anthropic()))

# Set per-node model
class CheapNode(Node):
    model = "gpt-4o-mini"  # Use cheap model for simple tasks

class AccurateNode(Node):
    model = "gpt-4o"  # Use expensive model for critical extractions
```

Set a budget to automatically stop when spending exceeds your limit:

```python
engine = AuraEngine(llm_client=OpenAI(), budget_usd=5.00)
```

---

## Cost Tracking

Monitor API spend per node and per model:

```python
report = engine.provider.cost_tracker.get_report()

print(f"Total: ${report['total_cost_usd']:.4f}")

for node, models in report["nodes"].items():
    for model, stats in models.items():
        print(f"  {node} ({model}): ${stats['cost_usd']:.4f} "
              f"({stats['calls']} calls, {stats['avg_latency_ms']}ms avg)")
```

---

## Consensus Extraction

Run the LLM multiple times and take the majority vote:

```python
class CriticalExtraction(Node):
    system_prompt = "Extract financial data. Accuracy is critical."
    extracts = FinancialData
    consensus = 3                       # Run 3 times
    consensus_strategy = "majority"     # Take majority vote
```

Strategies: `"majority"` (most common value wins) or `"unanimous"` (all must agree).

---

## Time-Travel Debugging

Every `engine.process()` call serializes its state to `.aura_trace/`. If something fails at step 5, fix your code and resume from step 4 — no need to re-run (and re-pay for) prior steps.

Trace files are saved as both JSON (human-readable) and pickle (resumable):

```
.aura_trace/
└── 20260301_131500/
    ├── step_001_ExtractLead.json
    ├── step_001_ExtractLead.pkl
    ├── step_002_QualifyBudget.json
    └── step_002_QualifyBudget.pkl
```

---

## Full Working Example

See [`examples/benchmark/`](examples/benchmark/) for a complete end-to-end example that processes 10 sales call transcripts through a 4-node pipeline, demonstrating all features working together.

```bash
# Run the synthetic benchmark (no API key needed)
python examples/benchmark/run_benchmark.py

# Run the live benchmark (requires OPENAI_API_KEY in .env)
python examples/benchmark/run_live.py --model gpt-4o-mini --runs 3
```
