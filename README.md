<p align="center">
  <h1 align="center">⚡ Aura-State</h1>
</p>

<p align="center">
  <strong>A Formally Verified LLM State Machine Compiler with Calibrated Uncertainty Guarantees.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.10+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="#benchmark-results"><img src="https://img.shields.io/badge/tests-65%20passing-brightgreen?style=for-the-badge" alt="65 Tests Passing"></a>
  <a href="#benchmark-results"><img src="https://img.shields.io/badge/Z3%20proofs-20%2F20-blueviolet?style=for-the-badge" alt="Z3 Proofs 20/20"></a>
</p>

<p align="center">
  <a href="docs/GUIDE.md">Usage Guide</a> · <a href="examples/benchmark/">Benchmarks</a> · <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

LLMs are chaotic. When you pass 50 messages to a standard chat wrapper, it hallucinates state transitions, fails math, and forgets context. LangChain wraps the chaos. CrewAI lets agents derail. **Aura-State compiles the chaos into mathematically rigid execution.**

```bash
pip install git+https://github.com/munshi007/Aura-State.git
```

---

## 💡 The Idea

**Aura-State starts from a different premise:** LLMs are powerful *extraction and reasoning engines*, but they should never be trusted with state management, mathematical computation, or workflow control. Those are formal problems with formal solutions.

> *If you can model your LLM workflow as a directed graph, you can prove properties about it before it ever runs — and you can verify every extraction after it completes. The gap between "it usually works" and "it provably works" is smaller than people think.*

### 🔬 Research Foundations

The framework draws from three research areas that, until now, have not been applied to LLM agent systems:

| Research Area | How We Use It | Origin |
|:---|:---|:---|
| 🏗️ **Model Checking** | Compile node graphs into Kripke structures and verify CTL properties before execution | Clarke, Emerson & Sistla (1986) — hardware & flight control verification |
| 📊 **Conformal Prediction** | Wrap LLM extractions in distribution-free confidence intervals with guaranteed coverage | Vovk, Gammerman & Shafer (2005) — calibrated uncertainty |
| 🧮 **SMT Solving** | Z3 theorem prover to formally verify extracted data satisfies business constraints | de Moura & Bjørner (2008) — Microsoft Research, Windows driver verification |

---

## ⚔️ Why Not Just Use LangChain?

| | LangChain | CrewAI | **Aura-State** |
|:---|:---|:---|:---|
| Architecture | Chain-of-calls | Free-form agents | **Compiled state machine** |
| Routing | Sequential | Agent decides | **MCTS + UCB1 scoring** |
| Math | LLM hallucinates | LLM hallucinates | **Sandboxed AST execution** |
| Verification | None | None | **Z3 theorem prover** |
| Confidence | None | None | **Conformal prediction intervals** |
| Workflow safety | Hope | Hope | **CTL model checking (proven)** |

---

## 🚀 Quick Start

**1. Define your nodes:**
```python
from aura_state import AuraEngine, Node, CompiledTransition
from pydantic import BaseModel, Field
from openai import OpenAI

class LeadData(BaseModel):
    name: str = Field(description="Full name")
    budget: int = Field(description="Budget in USD")
    timeline: str = Field(description="Buying timeline")

class ExtractLead(Node):
    system_prompt = "Extract lead info from a sales call transcript."
    extracts = LeadData

    def handle(self, user_text, extracted_data=None, memory=None):
        return "QualifyBudget", extracted_data.model_dump()

class QualifyBudget(Node):
    system_prompt = "Score the lead."
    sandbox_rule = "result = budget > 100000"

    def handle(self, user_text, extracted_data=None, memory=None):
        return "END", memory
```

**2. Wire and run:**
```python
engine = AuraEngine(llm_client=OpenAI())
engine.register(ExtractLead, QualifyBudget)
engine.connect([
    CompiledTransition(from_node=ExtractLead, to_node=QualifyBudget),
])

next_state, data = engine.process("ExtractLead", user_text="Hi, I'm Sarah. Budget is $450k.")
```

**3. Verify before you run:**
```python
from aura_state import verify_engine, reachability, eventual_completion

results = verify_engine(engine, [
    {"description": "QualifyBudget is reachable", "formula": reachability("QualifyBudget")},
    {"description": "All paths terminate", "formula": eventual_completion("QualifyBudget")},
])
# → PROVEN: QualifyBudget is reachable
# → PROVEN: All paths terminate
```

**4. Prove extractions are correct:**
```python
from aura_state import prove_extraction

result = prove_extraction(
    {"budget": 450000, "cost_per_sqft": 3, "total": 1350000},
    obligations=["budget > 0", "total == budget * cost_per_sqft"],
)
# result.verified = True — Z3 proves it, counterexample on failure
```

> 📖 **[Full Usage Guide →](docs/GUIDE.md)** — Detailed docs with code examples for every feature.

---

## 🧠 The 8 Innovations

### Core Engine

| # | Innovation | Algorithm | What It Does |
|:---|:---|:---|:---|
| 1 | 🔮 **Speculative Execution** | Thread pool pre-computation | Pre-computes likely next nodes in parallel |
| 2 | 📈 **Adaptive DAG** | Runtime health tracking | Monitors failure rates + auto-injects reflexion |
| 3 | 🔄 **Verification Loop** | Extract → verify → reflect → retry | Catches extraction errors before propagation |
| 4 | 🏗️ **Schema Compiler** | JSON Schema → Node classes | Compiles schemas into typed Nodes at runtime |
| 5 | 🔀 **Multi-Provider** | Per-node model routing | Routes each node to optimal model with failover |

### Formal Verification

| # | Innovation | Algorithm | What It Does |
|:---|:---|:---|:---|
| 6 | 🔒 **Temporal Logic** | Kripke + CTL model checking | Proves workflow safety *before* execution |
| 7 | 📊 **Conformal Prediction** | Split conformal prediction | Calibrated 95% CIs on every extracted field |
| 8 | 🧮 **Z3 Proof Engine** | SMT theorem proving | Proves data satisfies business constraints |

---

## 🏆 Benchmark Results

Live benchmark: **10 real-estate sales transcripts**, GPT-4o-mini, 30 API calls.

### 📋 Extraction Accuracy

```
Field             Correct   Total    Accuracy
───────────────  ────────  ──────  ──────────
name                   10      10       100%  ██████████
budget                 10      10       100%  ██████████
bedrooms               10      10       100%  ██████████
pre_approved            9      10        90%  █████████░
timeline                8      10        80%  ████████░░
city                    7      10        70%  ███████░░░
```

### ✅ Formal Verification

| Metric | Result |
|:---|:---|
| Temporal properties | **3/3 PROVEN** (reachability, mutual exclusion, termination) |
| Z3 proof obligations | **20/20 passed** on real LLM-extracted data |
| Conformal coverage | **100%** (all values within 95% CI) |
| Budget accuracy | **100% exact match** ($0 mean error) |

### ⚡ Performance

| Metric | Value |
|:---|:---|
| Avg latency | **1,277ms** |
| P50 latency | 1,193ms |
| P95 latency | 2,004ms |
| Total (30 calls) | 38.3s |

### 🔧 Run it yourself

```bash
# Synthetic benchmark (no API key needed)
python examples/benchmark/run_benchmark.py

# Live benchmark (requires OPENAI_API_KEY in .env)
python examples/benchmark/run_live.py --model gpt-4o-mini --runs 3
```

---

## 🏛️ How it works

```
User Input
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│                     AuraEngine                           │
│                                                          │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐  │
│  │ Adaptive DAG │───▶│ GraphRAG     │───▶│ Telepromp- │  │
│  │ Health Check │    │ Cache Lookup │    │ ter (KNN)  │  │
│  └─────────────┘    └──────────────┘    └────────────┘  │
│         │                                      │         │
│         ▼                                      ▼         │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐  │
│  │ Verification │───▶│   Node       │───▶│   MCTS     │  │
│  │ Loop (Z3)   │    │   handle()   │    │  Routing   │  │
│  └─────────────┘    └──────────────┘    └────────────┘  │
│         │                                      │         │
│         ▼                                      ▼         │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐  │
│  │  AuraTrace  │───▶│ Speculative  │───▶│   Output   │  │
│  │  (debug)    │    │  Execution   │    │            │  │
│  └─────────────┘    └──────────────┘    └────────────┘  │
└──────────────────────────────────────────────────────────┘
    │
    ▼
Next State + Extracted Data
```

---

## 📁 Architecture

```
aura_state/
├── core/
│   ├── engine.py              # AuraEngine — unified execution pipeline
│   ├── router.py              # MCTS routing with UCB1 scoring
│   ├── adaptive_graph.py      # Runtime DAG health monitoring
│   ├── verification_loop.py   # Extract → verify → reflect → retry
│   ├── providers.py           # Multi-provider LLM + cost tracking
│   └── exceptions.py
├── compiler/
│   ├── schema_compiler.py     # JSON Schema → Node class compilation
│   ├── json_generator.py      # Node → flow.json export
│   └── dspy_tuner.py          # KNN few-shot teleprompting
├── verification/
│   ├── temporal_verifier.py   # Kripke + CTL model checking
│   ├── conformal.py           # Split conformal prediction
│   └── proof_engine.py        # Z3 SMT constraint proofs
├── execution/
│   ├── tracer.py              # AuraTrace — time-travel debugging
│   └── sandbox.py             # AST-validated code execution
├── memory/
│   ├── trajectory_cache.py    # GraphRAG subgraph isomorphism
│   └── pruner.py              # Context window optimization
├── consensus/
│   └── auto_vote.py           # Multi-run extraction with voting
└── loaders/
    └── json_graph.py          # JSON/YAML → engine hydration
```

---

## 📦 Installation

```bash
pip install git+https://github.com/YOUR_USERNAME/aura-state.git
```

**Requirements:** Python 3.10+

**Dependencies:** `pydantic` · `instructor` · `openai` · `networkx` · `pyModelChecking` · `z3-solver` · `pyyaml`

---

## 🧪 Testing

```bash
python -m pytest tests/ -v
# 65 tests passing ✅
```

---

## 📚 Docs

| Resource | Description |
|:---|:---|
| 📖 [Usage Guide](docs/GUIDE.md) | Detailed how-to with code examples for every feature |
| 🤝 [Contributing](CONTRIBUTING.md) | Architecture overview and how to contribute |
| 🏆 [Benchmark](examples/benchmark/) | Synthetic and live benchmarks with observability |

---

## 📄 License

MIT
