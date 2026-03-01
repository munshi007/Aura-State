# Algorithms & Research Foundations

A deep-dive into every algorithm powering Aura-State, why we chose it, and how it maps to LLM orchestration.

---

## Table of Contents

- [CTL Model Checking](#ctl-model-checking-temporal-logic-verification)
- [Z3 SMT Solving](#z3-smt-solving-proof-engine)
- [Conformal Prediction](#conformal-prediction)
- [Monte Carlo Tree Search](#monte-carlo-tree-search-mcts-routing)
- [UCB1](#ucb1-upper-confidence-bound)
- [Subgraph Isomorphism](#subgraph-isomorphism-graphrag-cache)
- [KNN Few-Shot Teleprompting](#knn-few-shot-teleprompting)
- [AST Sandboxing](#ast-sandboxing-safe-math-execution)

---

## CTL Model Checking (Temporal Logic Verification)

**What it is:** CTL (Computation Tree Logic) is a branch of formal logic used to reason about all possible execution paths in a system. Model checking is the automated technique for verifying whether a system satisfies a CTL property.

**Origin:** Clarke, Emerson & Sistla, 1986 — *"Automatic Verification of Finite-State Concurrent Systems Using Temporal Logic Specifications."* This won the 2007 Turing Award. The technique is used to verify hardware circuits, network protocols, and flight control systems.

**How we use it:** Your node graph is compiled into a **Kripke structure** — a formal model `M = (S, S₀, R, L)` where:
- `S` = set of states (your nodes)
- `S₀` = initial states
- `R` = transition relation (your edges)
- `L` = labeling function (which node is active)

We then check CTL formulas against this structure:

| Formula | CTL Notation | Meaning |
|:---|:---|:---|
| Reachability | **EF(φ)** | "There exists a path where φ eventually holds" |
| Mutual Exclusion | **AG(¬(φ ∧ ψ))** | "On all paths, globally, φ and ψ are never both true" |
| Eventual Completion | **AF(φ)** | "On all paths, φ eventually holds" |
| Ordering | **A[¬ψ U φ]** | "On all paths, ψ doesn't hold until φ does" |

**Why not just check edges manually?** For simple linear graphs, you could. But the moment you have branching, loops, or conditional routing, the number of possible paths explodes. Model checking exhaustively verifies *every* path.

**Implementation:** We use [pyModelChecking](https://github.com/albertocasagrande/pyModelChecking), a Python library for CTL/LTL model checking, applied via `aura_state/verification/temporal_verifier.py`.

---

## Z3 SMT Solving (Proof Engine)

**What it is:** Z3 is a Satisfiability Modulo Theories (SMT) solver from Microsoft Research. Given a set of mathematical constraints, Z3 can prove whether they are satisfiable, unsatisfiable, or produce a counterexample.

**Origin:** de Moura & Bjørner, 2008 — *"Z3: An Efficient SMT Solver."* Used in Microsoft's Windows driver verification (SLAM project), in CompCert (verified C compiler), and in Dafny (verified programming language).

**How we use it:** After the LLM extracts data, we translate business rules into Z3 constraints and prove them:

```python
# LLM extracted: {budget: 450000, cost_per_sqft: 3, total: 1350000}
# Business rules: ["budget > 0", "total == budget * cost_per_sqft"]

# Z3 translates this to:
#   ∀ budget, cost_per_sqft, total:
#     budget = 450000 ∧ cost_per_sqft = 3 ∧ total = 1350000
#     → budget > 0 ∧ total = budget × cost_per_sqft
#
# Z3 result: SATISFIABLE (proven correct)
```

When a proof fails, Z3 produces a **counterexample** — the specific values that violate the constraint. This is not a heuristic; it's a mathematical proof.

**Why not just assert?** A Python `assert total == budget * cost_per_sqft` would catch mismatches, but Z3 can handle symbolic reasoning. It can prove that constraints are *always* satisfiable given a schema, or find edge cases that assertions would miss.

**Implementation:** `aura_state/verification/proof_engine.py` using [z3-solver](https://github.com/Z3Prover/z3).

---

## Conformal Prediction

**What it is:** A distribution-free statistical method that wraps point predictions with prediction intervals that have guaranteed coverage probability. Unlike Bayesian approaches, conformal prediction makes no assumptions about the data distribution.

**Origin:** Vovk, Gammerman & Shafer, 2005 — *"Algorithmic Learning in a Random World."* Based on exchangeability (a weaker assumption than i.i.d.).

**How we use it:** When the LLM extracts a numeric value (e.g., budget = $450,000), we run multiple extractions and compute a conformal interval:

```
Extractions: [$450k, $452k, $448k, $450k, $451k]

Split conformal method:
1. Calibration set: first 3 values
2. Test set: last 2 values
3. Nonconformity scores: |xi - median|
4. α = 0.05 (for 95% coverage)
5. Quantile of scores → interval width

Result: $450,000 ± $2,200 (95% CI: [$447,800, $452,200])
Coverage guarantee: ≥ 95% of true values fall within this interval
```

**Why not standard confidence intervals?** Standard CIs assume normality. LLM outputs are not normally distributed — they're discrete, multi-modal, and model-dependent. Conformal prediction is **distribution-free**: the coverage guarantee holds regardless of the underlying distribution.

**Implementation:** `aura_state/verification/conformal.py`.

---

## Monte Carlo Tree Search (MCTS Routing)

**What it is:** A search algorithm that stochastically samples execution paths to estimate high-reward branches. Traditionally, this involves running thousands of random simulations (rollouts).

**How we use it:** To maintain low latency and minimize token usage, Aura-State implements **Real-Time MCTS**. Instead of spawning fresh LLM simulations for every decision, we use the **AdaptiveDAG** as the "simulation memory." Every execution in production act as a simulation that updates the edge health.

When a node has multiple valid transitions, the engine queries the AdaptiveDAG for:
1. **Node Success Rate**: The historical reliability of the target node.
2. **Node Priors**: Does the node have Z3 proofs or sandboxing? (higher initial bias).
3. **Execution Count**: How many times have we explored this path?

This data is fed into the UCB1 formula to select the next state.

**Why not just let the LLM decide?** LLMs are stateless and have no concept of long-term reward or statistical confidence. MCTS provides a mathematical framework for making decisions that improve over time as more data is collected.

---

## UCB1 (Upper Confidence Bound)

**What it is:** The selection policy used within MCTS to balance exploitation (staying with a high-success path) and exploration (trying a less-visited path).

**Algorithm:**
Aura-State uses the standard UCB1 formula:
`Score = [SuccessRate + Priors] + [C * sqrt(ln(TotalVisits) / NodeVisits)]`

- **Exploitation term**: `SuccessRate` (from AdaptiveDAG) + `Priors` (node features).
- **Exploration term**: The standard UCB1 confidence interval.
- **C**: Exploration constant (default is √2).

**Behavior:**
- **Unvisited Paths**: Paths with 0 visits receive infinite exploration scores, forcing the system to test every edge at least once.
- **Convergence**: As total executions grow, the exploration bonus decays, and the system converges on the most reliable state transition paths.
- **Failure Shunning**: If the current execution trace already encountered a failure on a specific path, a local penalty is applied during that specific process call.

---

## Subgraph Isomorphism (GraphRAG Cache)

**What it is:** Given two graphs G and H, subgraph isomorphism determines whether G contains a subgraph structurally identical to H. This is an NP-complete problem in general, but efficient for small pattern graphs.

**Origin:** Ullmann, 1976 — *"An Algorithm for Subgraph Isomorphism."* Modern implementations use VF2 (Cordella et al., 2004).

**How we use it:** When a user query comes in, we extract entity-relationship triples into a small **pattern graph** and check if it's isomorphic to any subgraph in our **knowledge graph** (built from past successful executions):

```
User: "850 sqft room with vaulted ceilings"

Pattern graph:
  (room)--[has_area]--(850 sqft)
  (room)--[has_type]--(vaulted ceiling)

Knowledge graph (from past runs):
  (room_A)--[has_area]--(800 sqft)
  (room_A)--[has_type]--(vaulted ceiling)
  (room_A)--[cost]--(12500)
  ...

VF2 check: Is pattern ⊆ knowledge graph?
  → YES: return cached result, skip LLM call entirely
  → NO: proceed with LLM extraction
```

**Why not vector similarity?** Embedding-based similarity gives you "this is 87% similar" — a fuzzy answer. Subgraph isomorphism gives you a **mathematical guarantee**: the structure is either identical or it isn't. For caching, you want certainty, not probability.

**Implementation:** `aura_state/memory/trajectory_cache.py` using [NetworkX](https://networkx.org/) `is_isomorphic()`.

---

## KNN Few-Shot Teleprompting

**What it is:** A prompt optimization technique inspired by [DSPy](https://github.com/stanfordnlp/dspy) (Stanford NLP). Instead of manually writing few-shot examples, the system automatically selects the K most relevant past executions and injects them as demonstrations.

**Origin:** Khattab et al., 2023 — *"DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines."*

**How we use it:**

```
1. Record all successful (input, output) pairs per node
2. When a new input arrives:
   a. Embed the input
   b. KNN search against stored embeddings (K=3)
   c. Inject the top-K as few-shot examples in the prompt
3. The LLM sees: system prompt + 3 perfect examples + user input
```

**Why KNN over fixed examples?** Fixed few-shot examples are static — they may not be relevant to the current input. KNN dynamically selects the *most similar* past successes, giving the LLM maximally relevant demonstrations. This is particularly effective for nodes that handle diverse inputs.

**Implementation:** `aura_state/compiler/dspy_tuner.py`.

---

## AST Sandboxing (Safe Math Execution)

**What it is:** Python's `ast` module parses code into an Abstract Syntax Tree without executing it. We validate the tree against a whitelist of safe operations before execution.

**How we use it:** When a node has a `sandbox_rule` (e.g., `"result = wall_area * cost_per_sqft * 1.15"`), the engine:

```
1. Parse rule → AST
2. Walk the tree and verify:
   ✅ Arithmetic: +, -, *, /, **, %
   ✅ Comparisons: >, <, ==, !=
   ✅ Variables: only those in the extracted data
   ❌ Imports: blocked
   ❌ Function calls: blocked (no exec, eval, open, etc.)
   ❌ Attribute access: blocked (no os.system, etc.)
3. If safe → execute in isolated namespace
4. If unsafe → reject with error
```

**Why not just `eval()`?** `eval()` executes arbitrary Python code — an LLM could inject `__import__('os').system('rm -rf /')`. AST validation guarantees that only pre-approved operations execute. The LLM never touches the calculation; it only provides the variables.

**Implementation:** `aura_state/execution/sandbox.py`.

---

## References

1. Clarke, E.M., Emerson, E.A., & Sistla, A.P. (1986). Automatic verification of finite-state concurrent systems using temporal logic specifications. *ACM TOPLAS*.
2. de Moura, L., & Bjørner, N. (2008). Z3: An efficient SMT solver. *TACAS*.
3. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
4. Coulom, R. (2006). Efficient selectivity and backup operators in Monte-Carlo tree search. *CG*.
5. Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*.
6. Ullmann, J.R. (1976). An algorithm for subgraph isomorphism. *JACM*.
7. Khattab, O., et al. (2023). DSPy: Compiling declarative language model calls into self-improving pipelines. *Stanford NLP*.
8. Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*.
