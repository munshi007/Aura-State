# Contributing to Aura-State

## Getting Started

```bash
git clone <repo-url>
cd aura-state
pip install -e .
python -m pytest tests/ -v   # 65 tests should pass
```

## Architecture

```
aura_state/
├── core/           → Engine, MCTS router, adaptive DAG, verification loop, providers
├── compiler/       → Schema compiler, JSON generator, DSPy-inspired teleprompting
├── verification/   → Temporal verifier (CTL), conformal prediction, Z3 proof engine
├── execution/      → AuraTrace debugger, AST sandbox
├── memory/         → GraphRAG cache, context pruner
├── consensus/      → Multi-run extraction with voting
└── loaders/        → JSON/YAML graph loader
```

## Running Benchmarks

```bash
# Synthetic (no API key)
python examples/benchmark/run_benchmark.py

# Live (requires OPENAI_API_KEY in .env)
python examples/benchmark/run_live.py --model gpt-4o-mini --runs 3
```

## Pull Requests

1. Create a feature branch.
2. Write tests for your changes.
3. Ensure `python -m pytest tests/ -v` passes (all 65 tests).
4. Open a PR with a clear description of *what* and *why*.

## Design Principles

- Every feature uses a real algorithm, not an API wrapper.
- `AuraEngine` is the single entrypoint. All internals are always active.
- Math is executed in a sandboxed AST interpreter, never by the LLM.
- The DAG is the source of truth for state transitions.
- Formal verification happens *before* execution, not after.
