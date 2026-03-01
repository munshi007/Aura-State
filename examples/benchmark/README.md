# Aura-State Benchmark: Real Estate Lead Qualifier

This benchmark demonstrates **all 8 innovations** in a single end-to-end pipeline.
No API keys required — uses mocked LLM responses to show exactly what happens
at each stage.

## What It Does

Processes a raw sales call transcript through a 4-node pipeline:

```
Transcript → ExtractLead → QualifyBudget → VerifyData → RouteDecision
```

At each step, different innovations fire:

| Stage | Innovation Used | What It Proves |
|-------|----------------|----------------|
| Before execution | Temporal Verifier | Workflow is mathematically safe |
| ExtractLead | Verification Loop, Conformal Prediction | Data is verified + confidence intervals |
| QualifyBudget | Sandbox Interpreter | Math is deterministic, not LLM-hallucinated |
| VerifyData | Z3 Proof Engine | Extracted data formally satisfies business rules |
| RouteDecision | MCTS Routing | Transition is scored, not guessed |
| Every step | Adaptive DAG, Speculative Execution, Cost Tracking | Graph health, parallel pre-compute, budget |

## Run It

```bash
cd examples/benchmark
python run_benchmark.py
```

## Files

- `nodes.py` — The 4 Node classes (what the developer writes)
- `dataset.py` — 10 synthetic sales transcripts with known ground truth
- `run_benchmark.py` — Runs the full pipeline and prints the results table
