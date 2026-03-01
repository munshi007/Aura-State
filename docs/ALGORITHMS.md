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
- [AST Sandboxing](#ast-sandboxing-neurosymbolic-math)

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

**What it is:** A search algorithm that uses random simulations to evaluate decision trees. Instead of exhaustively searching all paths (infeasible for large graphs), MCTS samples paths stochastically and builds an asymmetric tree focused on the most promising branches.

**Origin:** Coulom, 2006 — *"Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search."* Famously used in AlphaGo (Silver et al., 2016) to beat the world Go champion.

**How we use it:** When a node has multiple valid transitions and the node's `handle()` returns an ambiguous target, MCTS simulates forward paths:

```
Current: ReviewNode
Possible transitions: [ApproveNode, RejectNode, EscalateNode]

MCTS iterations:
  Simulate: ReviewNode → ApproveNode → FinalizeNode → END    → reward: 0.82
  Simulate: ReviewNode → RejectNode → NotifyNode → END       → reward: 0.45
  Simulate: ReviewNode → EscalateNode → ManagerNode → END    → reward: 0.63
  Simulate: ReviewNode → ApproveNode → FinalizeNode → END    → reward: 0.79
  ...

After N simulations: ApproveNode has highest average reward → selected
```

**Why not just let the LLM decide?** LLMs are stateless text generators. They have no memory of which paths worked before, no concept of long-term reward, and no mathematical framework for exploration vs. exploitation. MCTS provides all three.

**Implementation:** `aura_state/core/router.py`, invoked by `_mcts_select()` in `engine.py`.

---

## UCB1 (Upper Confidence Bound)

**What it is:** A bandit algorithm that balances exploitation (choosing the best-known option) with exploration (trying less-tested options). UCB1 is the selection policy used within MCTS to decide which branch to explore next.

**Origin:** Auer, Cesa-Bianchi & Fischer, 2002 — *"Finite-time Analysis of the Multiarmed Bandit Problem."*

**Formula:**

```
UCB1(i) = X̄ᵢ + C × √(ln(N) / nᵢ)

Where:
  X̄ᵢ  = average reward of branch i (exploitation)
  N   = total number of simulations
  nᵢ  = number of times branch i was visited
  C   = exploration constant (typically √2)
```

**Intuition:**
- A branch with high `X̄ᵢ` (good track record) gets a high score → exploitation
- A branch with low `nᵢ` (rarely visited) gets a bonus from the √ term → exploration
- As `N` grows, the exploration bonus shrinks → converges to optimal

**Why UCB1 over random?** Random selection has no memory. Epsilon-greedy is simplistic. UCB1 is **provably optimal** — it achieves logarithmic regret, meaning it converges to the best action with minimal wasted exploration.

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

## AST Sandboxing (Neurosymbolic Math)

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
