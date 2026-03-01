#!/usr/bin/env python3
"""
Aura-State Synthetic Benchmark
==============================

Runs 10 real-estate lead qualification transcripts through the full
Aura-State pipeline with mocked LLM responses, demonstrating all 8
innovations working together.

No API key needed. Run:
    python examples/benchmark/run_benchmark.py
"""
import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from aura_state import (
    AuraEngine, CompiledTransition,
    compile_kripke, verify_engine,
    reachability, always_before, mutual_exclusion, eventual_completion,
    PropertyResult,
    conformal_interval, conformal_from_extractions,
    prove_extraction, prove_consistency,
)
from nodes import ExtractLead, QualifyBudget, VerifyData, RouteLead
from dataset import TRANSCRIPTS


def separator(title):
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def sub_separator(title):
    print(f"\n  --- {title} ---")


def run():
    # ================================================================
    # STEP 1: Build the pipeline
    # ================================================================
    separator("STEP 1: Building the Pipeline")

    engine = AuraEngine()
    engine.register(ExtractLead, QualifyBudget, VerifyData, RouteLead)
    engine.connect([
        CompiledTransition(from_node=ExtractLead, to_node=QualifyBudget),
        CompiledTransition(from_node=QualifyBudget, to_node=VerifyData),
        CompiledTransition(from_node=VerifyData, to_node=RouteLead),
    ])

    print("\n  Registered 4 nodes:")
    for name in engine._nodes:
        node = engine._nodes[name]
        has_ext = "yes" if node.extracts else "no"
        has_sandbox = "yes" if node.sandbox_rule else "no"
        print(f"    {name:20s}  extracts={has_ext:3s}  sandbox={has_sandbox}")

    print(f"\n  Transitions:")
    for src, targets in engine._transitions.items():
        for tgt in targets:
            print(f"    {src} --> {tgt}")

    # ================================================================
    # STEP 2: Temporal Logic Verification
    # Prove workflow properties BEFORE anything runs
    # ================================================================
    separator("STEP 2: Temporal Logic Verification")
    print("\n  Compiling node graph to Kripke structure...")

    properties = [
        {
            "description": "RouteLead is reachable from the start",
            "formula": reachability("RouteLead"),
        },
        {
            "description": "Cannot be in QualifyBudget and RouteLead simultaneously",
            "formula": mutual_exclusion("QualifyBudget", "RouteLead"),
        },
        {
            "description": "Every path reaches a terminal (RouteLead)",
            "formula": eventual_completion("RouteLead"),
        },
    ]

    results = verify_engine(engine, properties)
    for r in results:
        status = "PROVEN" if r.result == PropertyResult.PROVEN else "VIOLATED"
        icon = "+" if status == "PROVEN" else "X"
        print(f"  [{icon}] {status:8s}  {r.property_text}")
        if r.satisfying_states:
            print(f"             Satisfying: {r.satisfying_states}")

    # ================================================================
    # STEP 3: Process all 10 transcripts
    # ================================================================
    separator("STEP 3: Processing 10 Transcripts")

    all_results = []
    correct_routes = 0
    total_z3_proofs = 0
    total_z3_pass = 0
    budget_extractions = []

    for entry in TRANSCRIPTS:
        tid = entry["id"]
        gt = entry["ground_truth"]
        sub_separator(f"Transcript #{tid}: {gt['name']}")

        # Simulate the extraction (mocked — in production the LLM does this)
        extracted = {
            "name": gt["name"],
            "budget": gt["budget"],
            "bedrooms": gt["bedrooms"],
            "city": gt["city"],
            "timeline": gt["timeline"],
            "pre_approved": gt["pre_approved"],
        }
        budget_extractions.append(gt["budget"])

        print(f"    Extracted: budget=${gt['budget']:,}  bedrooms={gt['bedrooms']}  "
              f"timeline={gt['timeline']}  pre_approved={gt['pre_approved']}")

        # --- Z3 Proof Engine ---
        proof = prove_extraction(extracted, [
            "budget > 0",
            "bedrooms >= 0",
        ])
        total_z3_proofs += 2
        total_z3_pass += (2 - len(proof.failed_obligations))
        proof_icon = "+" if proof.verified else "X"
        print(f"    Z3 Proof:  [{proof_icon}] {'PASS' if proof.verified else 'FAIL'}"
              f"  (obligations: budget>0, bedrooms>=0)")

        # --- Run the deterministic pipeline (QualifyBudget + VerifyData + Route) ---
        qualify = QualifyBudget()
        _, qual_data = qualify.handle("", memory=extracted)

        verify = VerifyData()
        _, ver_data = verify.handle("", memory=qual_data)

        route_node = RouteLead()
        _, final_data = route_node.handle("", memory=ver_data)

        predicted_route = final_data["route"]
        expected_route = entry["expected_route"]
        match = predicted_route == expected_route
        if match:
            correct_routes += 1

        match_icon = "+" if match else "X"
        print(f"    Route:     [{match_icon}] predicted={predicted_route}  "
              f"expected={expected_route}")
        print(f"    Scores:    urgency={final_data.get('urgency_score')}  "
              f"readiness={final_data.get('readiness_score')}  "
              f"budget/bed=${final_data.get('budget_per_bedroom', 0):,.0f}")

        all_results.append({
            "id": tid,
            "name": gt["name"],
            "budget": gt["budget"],
            "predicted_route": predicted_route,
            "expected_route": expected_route,
            "correct": match,
            "z3_passed": proof.verified,
        })

    # ================================================================
    # STEP 4: Conformal Prediction
    # ================================================================
    separator("STEP 4: Conformal Prediction on Budget Extractions")

    ci = conformal_interval(
        [float(b) for b in budget_extractions],
        confidence=0.95,
    )
    print(f"\n  Sample size:     {ci.n_samples} extractions")
    print(f"  Point estimate:  ${ci.point_estimate:,.0f}")
    print(f"  95% CI:          [${ci.lower:,.0f}, ${ci.upper:,.0f}]")
    print(f"  Interpretation:  If we extract 'budget' from a new transcript,")
    print(f"                   we are 95% confident it falls in this range.")

    # Show per-value check
    in_range = sum(1 for b in budget_extractions if ci.lower <= b <= ci.upper)
    print(f"\n  Coverage check:  {in_range}/{len(budget_extractions)} values "
          f"fall within the interval ({in_range/len(budget_extractions)*100:.0f}%)")

    # ================================================================
    # STEP 5: Adaptive DAG Health
    # ================================================================
    separator("STEP 5: Adaptive DAG Health Report")

    dag = engine.adaptive_graph
    for name in engine._nodes:
        for _ in range(10):
            dag.record_execution(name, success=True, latency_ms=50.0)
        dag.record_execution(name, success=False, latency_ms=200.0)

    report = dag.get_health_report()
    print(f"\n  {'Node':20s}  {'Executions':>10s}  {'Fail Rate':>10s}  {'Avg Latency':>12s}")
    print(f"  {'─' * 20}  {'─' * 10}  {'─' * 10}  {'─' * 12}")
    for node_name, metrics in report.items():
        print(f"  {node_name:20s}  {metrics['total_executions']:10d}  "
              f"{metrics['fail_rate']:9.1%}  {metrics['avg_latency_ms']:10.1f}ms")

    # ================================================================
    # STEP 6: Cost Tracking
    # ================================================================
    separator("STEP 6: Cost Tracking Report")

    cost_tracker = engine.provider.cost_tracker
    for i, name in enumerate(engine._nodes):
        cost_tracker.record(
            node_name=name, model="gpt-4o",
            input_tokens=500, output_tokens=200,
            latency_ms=80.0 + i * 10,
        )

    report = cost_tracker.get_report()
    budget = report.get("budget_usd")
    budget_str = f"${budget:.2f}" if budget else "No limit"
    print(f"\n  Total API cost:   ${report['total_cost_usd']:.4f}")
    print(f"  Budget:           {budget_str}")

    total_calls = 0
    total_tokens = 0
    if report["nodes"]:
        print(f"\n  Per-node breakdown:")
        for node_name, models in report["nodes"].items():
            for model_name, stats in models.items():
                total_calls += stats["calls"]
                total_tokens += stats["input_tokens"] + stats["output_tokens"]
                print(f"    {node_name:20s}  {model_name:10s}  "
                      f"${stats['cost_usd']:.4f}  "
                      f"({stats['calls']} calls, "
                      f"{stats['input_tokens'] + stats['output_tokens']:,} tokens, "
                      f"{stats['avg_latency_ms']}ms avg)")

    # ================================================================
    # STEP 7: Summary
    # ================================================================
    separator("BENCHMARK SUMMARY")

    print(f"""
  Pipeline:          4 nodes, 3 transitions
  Transcripts:       {len(TRANSCRIPTS)} processed

  ROUTING ACCURACY
  ────────────────
  Correct routes:    {correct_routes}/{len(TRANSCRIPTS)} ({correct_routes/len(TRANSCRIPTS)*100:.0f}%)

  Z3 FORMAL PROOFS
  ────────────────
  Obligations:       {total_z3_proofs} checked
  Passed:            {total_z3_pass}/{total_z3_proofs} ({total_z3_pass/total_z3_proofs*100:.0f}%)

  TEMPORAL VERIFICATION
  ─────────────────────
  Properties checked: {len(properties)}
  Proven:            {sum(1 for r in results if r.result == PropertyResult.PROVEN)}/{len(properties)}

  CONFORMAL PREDICTION
  ────────────────────
  95% CI on budget:  [${ci.lower:,.0f}, ${ci.upper:,.0f}]
  Coverage:          {in_range}/{len(budget_extractions)} ({in_range/len(budget_extractions)*100:.0f}%)

  COST TRACKING
  ─────────────
  Total spend:       ${report['total_cost_usd']:.4f}
""")

    print(f"  {'ID':>3s}  {'Name':20s}  {'Budget':>10s}  {'Route':6s}  {'Expected':8s}  {'Match':5s}  {'Z3':3s}")
    print(f"  {'─'*3}  {'─'*20}  {'─'*10}  {'─'*6}  {'─'*8}  {'─'*5}  {'─'*3}")
    for r in all_results:
        match_mark = "yes" if r["correct"] else "NO"
        z3_mark = "yes" if r["z3_passed"] else "NO"
        print(f"  {r['id']:3d}  {r['name']:20s}  ${r['budget']:>9,}  "
              f"{r['predicted_route']:6s}  {r['expected_route']:8s}  "
              f"{match_mark:5s}  {z3_mark:3s}")

    print()


if __name__ == "__main__":
    run()
