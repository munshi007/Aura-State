# Aura-State Benchmark: Real Estate Lead Qualifier

This benchmark demonstrates the full end-to-end pipeline with formal verification, Thompson-bandit routing, and sandboxed math.
No API keys required — uses mocked LLM responses by default to show exactly what happens at each stage.

## What It Does

Processes a raw sales call transcript through a 4-node pipeline:

```
Transcript → ExtractLead → QualifyBudget → VerifyData → RouteDecision
```

At each step, different verification and routing features fire:

| Stage | Mechanism | What It Proves |
|-------|-----------|----------------|
| Before execution | Temporal Verifier | Workflow is mathematically safe (CTL) |
| ExtractLead | Verification Loop | Data is verified + conformal confidence intervals |
| QualifyBudget | Sandbox Interpreter | Math is deterministic, not LLM-hallucinated |
| VerifyData | Z3 Proof Engine | Extracted data formally satisfies business rules |
| RouteDecision | Thompson Bandit | Fallback edge is sampled from per-edge Beta-Bernoulli posteriors |
| Every step | Adaptive DAG | Runtime health metrics + per-edge routing posteriors |

## Run It

```bash
python run_benchmark.py
```

## Files

- `nodes.py` — The 4 Node classes (what the developer writes)
- `dataset.py` — 10 synthetic sales transcripts with known ground truth
- `run_benchmark.py` — Runs the full pipeline and prints the results table
