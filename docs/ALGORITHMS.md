# Algorithms & Research Foundations

A deep-dive into every algorithm powering Aura-State, why we chose it, and how it maps to LLM orchestration.

---

## Table of Contents

- [CTL Model Checking](#ctl-model-checking-temporal-logic-verification)
- [Z3 SMT Solving](#z3-smt-solving-proof-engine)
- [Conformal Prediction](#conformal-prediction)
- [Thompson Sampling (Bandit Router)](#thompson-sampling-bandit-router)
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

**Origin:** Vovk, Gammerman & Shafer, 2005 — *"Algorithmic Learning in a Random World."* Based on exchangeability (a weaker assumption than i.i.d.). Our estimator is the **jackknife+** variant of Barber, Candès, Ramdas & Tibshirani, 2021 — *"Predictive inference with the jackknife+."*

**How we use it:** When the LLM extracts a numeric value (e.g., budget = $450,000), we run multiple extractions and compute a jackknife+ interval:

```
Extractions: [$450k, $452k, $448k, $450k, $451k, ...]

Jackknife+ method (Barber et al. 2021):
1. For each i, leave it out and compute μ₋ᵢ = median of the rest
2. LOO nonconformity score: Rᵢ = |vᵢ − μ₋ᵢ|
3. α = 0.05 (for 95% coverage); rank k = ceil((1 − α)·(n + 1))
4. upper = k-th smallest of {μ₋ᵢ + Rᵢ}, lower = (n+1−k)-th smallest of {μ₋ᵢ − Rᵢ}
   (order statistics, NOT interpolated quantiles)

Result: $450,000 ± $2,200 (95% CI: [$447,800, $452,200])
Coverage guarantee: covers a fresh draw with probability ≥ 1 − 2α worst-case,
~ 1 − α empirically for well-behaved data
```

Jackknife+ uses every sample via leave-one-out rather than wasting half on a
disjoint calibration fold, which matters because N here is just a handful of
consensus runs per field. Below the minimum sample count
(`n ≥ ceil(1/α) − 1`, = 19 at 95%) there is no valid finite-sample threshold:
the interval falls back to the raw min..max range and is flagged **uncalibrated**
(`confidence = None`) — we never stamp a nominal coverage label we can't back up.
Note the interval measures the model's run-to-run **dispersion / self-agreement**,
not its error against an external ground truth.

**Why not standard confidence intervals?** Standard CIs assume normality. LLM outputs are not normally distributed — they're discrete, multi-modal, and model-dependent. Conformal prediction is **distribution-free**: the coverage guarantee holds regardless of the underlying distribution.

**Implementation:** `aura_state/verification/conformal.py`.

---

## Thompson Sampling (Bandit Router)

**What it is:** A Bayesian multi-armed bandit strategy. Each arm (here, each outgoing edge) carries a posterior distribution over its success probability. To choose, you draw one sample from every arm's posterior and pick the arm with the highest sample. High-uncertainty arms occasionally sample high and get tried; arms with a proven track record are picked most of the time. Exploration falls out of the posterior variance — there is no exploration constant to tune.

**Origin:** Thompson, 1933 — *"On the Likelihood that One Unknown Probability Exceeds Another in View of the Evidence of Two Samples."* The Beta-Bernoulli conjugacy that makes the update a one-line increment is the same result.

**How we use it:** Routing is a **fallback only**. Normally a node's `handle()` returns the name of the next node, which must be a declared transition. If `handle()` returns an edge that is *not* a declared transition, the engine falls back to the bandit router to pick a feasible edge:

1. **Filter to CTL-feasible edges first.** Only transitions that the graph's temporal model admits are candidates — the router never invents an unreachable jump.
2. **Sample each edge's posterior.** Every edge keeps a Beta-Bernoulli posterior in `EdgeStats` (`aura_state/core/adaptive_graph.py`): `Beta(α, β)` where α tracks successes and β tracks failures. We draw `θ_e ~ Beta(α_e, β_e)` for each candidate edge.
3. **Argmax over samples.** The edge with the largest sampled `θ_e` is selected (`_route_select` in `engine.py`).
4. **Update.** The Bernoulli reward (success in [0,1]) increments α on success and β on failure, so the posterior sharpens toward the truth over time.

To stay responsive under **non-stationarity** (a node's reliability drifting over time), the posterior counts are **discounted** — older observations decay, so recent evidence dominates.

**Why no exploration constant?** UCB-style policies bolt an explicit exploration bonus onto a point estimate and require tuning a constant `C`. Thompson sampling explores *proportionally to its uncertainty* by construction: a wide posterior naturally produces occasional high draws, and as evidence accumulates the posterior narrows and exploration fades on its own. Nothing to tune, and it matches or beats UCB1 empirically on Bernoulli bandits.

**Implementation:** `EdgeStats` posteriors in `aura_state/core/adaptive_graph.py`; selection in `_route_select` in `engine.py`.

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
4. Barber, R.F., Candès, E.J., Ramdas, A., & Tibshirani, R.J. (2021). Predictive inference with the jackknife+. *Annals of Statistics* 49(1): 486-507.
5. Thompson, W.R. (1933). On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. *Biometrika* 25(3-4): 285-294.
6. Khattab, O., et al. (2023). DSPy: Compiling declarative language model calls into self-improving pipelines. *Stanford NLP*.
