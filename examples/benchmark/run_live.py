#!/usr/bin/env python3
"""
Aura-State Live Benchmark
=========================

Runs real OpenAI API calls against 10 sales transcripts, then measures
extraction accuracy, Z3 proof results, conformal prediction intervals,
and total cost.

Requires OPENAI_API_KEY in environment or .env file.

Usage:
    python examples/benchmark/run_live.py
    python examples/benchmark/run_live.py --runs 3    # consensus runs per transcript
    python examples/benchmark/run_live.py --model gpt-4o-mini
"""
import sys
import os
import time
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
from openai import OpenAI
import instructor

from aura_state import (
    AuraEngine, Node, CompiledTransition,
    compile_kripke, verify_engine,
    reachability, mutual_exclusion, eventual_completion,
    PropertyResult,
    conformal_interval, conformal_from_extractions,
    prove_extraction,
)
from pydantic import BaseModel, Field
from dataset import TRANSCRIPTS


load_dotenv()


# -- Pydantic schema for extraction --

class LeadData(BaseModel):
    name: str = Field(description="Full name of the lead from the transcript")
    budget: int = Field(description="Total housing budget in USD")
    bedrooms: int = Field(description="Number of bedrooms requested, 0 if not mentioned")
    city: str = Field(description="Preferred city or area, 'unknown' if not mentioned")
    timeline: str = Field(description="One of: immediate, 1-3 months, 3-6 months, exploring")
    pre_approved: bool = Field(description="Whether the lead has mortgage pre-approval or cash ready")


# -- Nodes --

class ExtractLead(Node):
    system_prompt = (
        "Extract structured lead information from a real estate sales call transcript. "
        "Be precise with numbers. If the timeline is unclear, classify as 'exploring'."
    )
    extracts = LeadData

    def handle(self, user_text, extracted_data=None, memory=None):
        data = extracted_data.model_dump() if extracted_data else {}
        return "QualifyBudget", data


class QualifyBudget(Node):
    system_prompt = "Calculate qualification metrics."

    def handle(self, user_text, extracted_data=None, memory=None):
        data = memory or {}
        budget = data.get("budget", 0)
        bedrooms = data.get("bedrooms", 1) or 1
        timeline = data.get("timeline", "exploring")
        urgency_map = {"immediate": 10, "1-3 months": 7, "3-6 months": 4, "exploring": 1}
        readiness = 8 if data.get("pre_approved") else 3

        data["budget_per_bedroom"] = round(budget / bedrooms, 2) if bedrooms > 0 else float(budget)
        data["urgency_score"] = urgency_map.get(timeline, 1)
        data["readiness_score"] = readiness
        return "VerifyData", data


class VerifyData(Node):
    system_prompt = "Verify extracted data."

    def handle(self, user_text, extracted_data=None, memory=None):
        data = memory or {}
        data["data_valid"] = data.get("budget", 0) > 0 and data.get("name", "") != ""
        return "RouteLead", data


class RouteLead(Node):
    system_prompt = "Route the lead."

    def handle(self, user_text, extracted_data=None, memory=None):
        data = memory or {}
        urgency = data.get("urgency_score", 1)
        readiness = data.get("readiness_score", 1)
        combined = urgency + readiness

        if combined >= 15:
            route, reason = "hot", f"urgency={urgency} + readiness={readiness}"
        elif combined >= 8:
            route, reason = "warm", f"urgency={urgency} + readiness={readiness}"
        else:
            route, reason = "cold", f"urgency={urgency} + readiness={readiness}"

        data["route"] = route
        data["reason"] = reason
        return "END", data


def separator(title):
    w = 74
    print(f"\n{'=' * w}")
    print(f"  {title}")
    print(f"{'=' * w}")


def field_match(extracted, truth, field):
    ev = extracted.get(field)
    tv = truth.get(field)
    if isinstance(tv, (int, float)):
        return abs(ev - tv) / max(abs(tv), 1) < 0.15 if isinstance(ev, (int, float)) else False
    if isinstance(tv, bool):
        return ev == tv
    if isinstance(tv, str):
        return str(ev).lower().strip() == str(tv).lower().strip()
    return ev == tv


def run(model: str, runs_per_transcript: int):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    openai_client = OpenAI(api_key=api_key)
    client = instructor.from_openai(openai_client)

    separator(f"LIVE BENCHMARK — model={model}, runs={runs_per_transcript}")
    print(f"\n  Using real OpenAI API calls. Cost depends on model and token usage.")

    # -- Build engine --
    engine = AuraEngine(llm_client=openai_client)
    engine.register(ExtractLead, QualifyBudget, VerifyData, RouteLead)
    engine.connect([
        CompiledTransition(from_node=ExtractLead, to_node=QualifyBudget),
        CompiledTransition(from_node=QualifyBudget, to_node=VerifyData),
        CompiledTransition(from_node=VerifyData, to_node=RouteLead),
    ])

    # -- Step 1: Temporal verification (before any API calls) --
    separator("STEP 1: Pre-Flight Temporal Verification")
    props = [
        {"description": "RouteLead is reachable", "formula": reachability("RouteLead")},
        {"description": "Mutual exclusion: QualifyBudget vs RouteLead", "formula": mutual_exclusion("QualifyBudget", "RouteLead")},
        {"description": "All paths reach terminal", "formula": eventual_completion("RouteLead")},
    ]
    for r in verify_engine(engine, props):
        status = "PROVEN" if r.result == PropertyResult.PROVEN else "VIOLATED"
        print(f"  [{'+' if status == 'PROVEN' else 'X'}] {status:8s}  {r.property_text}")

    # -- Step 2: Live extractions --
    separator("STEP 2: Live LLM Extractions")

    all_extractions = []
    all_timings = []
    field_accuracy = {f: {"correct": 0, "total": 0} for f in ["name", "budget", "bedrooms", "city", "timeline", "pre_approved"]}
    correct_routes = 0
    z3_pass = 0
    z3_total = 0
    total_tokens_in = 0
    total_tokens_out = 0

    for entry in TRANSCRIPTS:
        tid = entry["id"]
        gt = entry["ground_truth"]
        transcript = entry["transcript"]

        print(f"\n  --- Transcript #{tid}: {gt['name']} ---")

        # Do N extraction runs for conformal prediction
        run_extractions = []
        run_timings = []

        for run_idx in range(runs_per_transcript):
            t0 = time.time()
            try:
                result = client.chat.completions.create(
                    model=model,
                    response_model=LeadData,
                    messages=[
                        {"role": "system", "content": ExtractLead.system_prompt},
                        {"role": "user", "content": transcript},
                    ],
                    max_retries=2,
                )
                elapsed_ms = (time.time() - t0) * 1000
                run_extractions.append(result)
                run_timings.append(elapsed_ms)

                if run_idx == 0:
                    ext = result.model_dump()
                    print(f"    Extracted: name={ext['name']}, budget=${ext['budget']:,}, "
                          f"beds={ext['bedrooms']}, city={ext['city']}")
                    print(f"    Timeline: {ext['timeline']}, pre_approved={ext['pre_approved']}")
                    print(f"    Latency:  {elapsed_ms:.0f}ms")

            except Exception as e:
                elapsed_ms = (time.time() - t0) * 1000
                print(f"    [Run {run_idx+1}] ERROR: {e} ({elapsed_ms:.0f}ms)")
                continue

        if not run_extractions:
            print(f"    SKIPPED — all extraction runs failed")
            continue

        primary = run_extractions[0].model_dump()
        all_extractions.append({"id": tid, "extracted": primary, "ground_truth": gt})
        all_timings.extend(run_timings)

        # Field-level accuracy
        for field in field_accuracy:
            field_accuracy[field]["total"] += 1
            if field_match(primary, gt, field):
                field_accuracy[field]["correct"] += 1

        # Z3 proof on the real extraction
        proof = prove_extraction(primary, ["budget > 0", "bedrooms >= 0"])
        z3_total += 2
        z3_pass += (2 - len(proof.failed_obligations))
        print(f"    Z3 Proof:  {'PASS' if proof.verified else 'FAIL'}")

        # Route through pipeline
        qualify = QualifyBudget()
        _, qual_data = qualify.handle("", memory=primary)
        verify = VerifyData()
        _, ver_data = verify.handle("", memory=qual_data)
        route_node = RouteLead()
        _, final_data = route_node.handle("", memory=ver_data)

        predicted = final_data["route"]
        expected = entry["expected_route"]
        match = predicted == expected
        if match:
            correct_routes += 1
        print(f"    Route:     predicted={predicted}  expected={expected}  "
              f"{'MATCH' if match else 'MISMATCH'}")

        # Conformal prediction (if multiple runs)
        if len(run_extractions) >= 3:
            cp_result = conformal_from_extractions(run_extractions, confidence=0.95)
            if "budget" in cp_result.intervals:
                iv = cp_result.intervals["budget"]
                print(f"    Conformal: budget 95% CI = [${iv.lower:,.0f}, ${iv.upper:,.0f}]")

    # -- Step 3: Aggregate results --
    separator("STEP 3: Field-Level Accuracy")
    print(f"\n  {'Field':15s}  {'Correct':>8s}  {'Total':>6s}  {'Accuracy':>10s}")
    print(f"  {'─'*15}  {'─'*8}  {'─'*6}  {'─'*10}")
    for field, stats in field_accuracy.items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            bar = "█" * int(acc * 10) + "░" * (10 - int(acc * 10))
            print(f"  {field:15s}  {stats['correct']:8d}  {stats['total']:6d}  {acc:9.0%}  {bar}")

    # -- Step 4: Conformal prediction over all budgets --
    separator("STEP 4: Conformal Prediction (All Budgets)")
    budgets = [e["extracted"]["budget"] for e in all_extractions if "budget" in e["extracted"]]
    gt_budgets = [e["ground_truth"]["budget"] for e in all_extractions]

    if budgets:
        ci = conformal_interval([float(b) for b in budgets], confidence=0.95)
        print(f"\n  Extracted budgets:  {[f'${b:,}' for b in budgets]}")
        print(f"  Point estimate:    ${ci.point_estimate:,.0f}")
        print(f"  95% CI:            [${ci.lower:,.0f}, ${ci.upper:,.0f}]")

        budget_errors = [abs(e - g) for e, g in zip(budgets, gt_budgets)]
        mean_error = sum(budget_errors) / len(budget_errors) if budget_errors else 0
        exact_match = sum(1 for e in budget_errors if e == 0)
        print(f"\n  Budget accuracy:")
        print(f"    Exact match:     {exact_match}/{len(budgets)} ({exact_match/len(budgets)*100:.0f}%)")
        print(f"    Mean abs error:  ${mean_error:,.0f}")

    # -- Step 5: Timing report --
    separator("STEP 5: Latency Report")
    if all_timings:
        avg = sum(all_timings) / len(all_timings)
        p50 = sorted(all_timings)[len(all_timings) // 2]
        p95 = sorted(all_timings)[int(len(all_timings) * 0.95)]
        total_s = sum(all_timings) / 1000
        print(f"\n  Total API calls:   {len(all_timings)}")
        print(f"  Total wall time:   {total_s:.1f}s")
        print(f"  Avg latency:       {avg:.0f}ms")
        print(f"  P50 latency:       {p50:.0f}ms")
        print(f"  P95 latency:       {p95:.0f}ms")

    # -- Step 6: Summary --
    separator("FINAL SUMMARY")
    n = len(all_extractions)
    print(f"""
  Model:             {model}
  Runs per transcript: {runs_per_transcript}
  Transcripts:       {n} processed

  ROUTING ACCURACY
  ────────────────
  Correct routes:    {correct_routes}/{n} ({correct_routes/n*100:.0f}%)

  Z3 FORMAL PROOFS
  ────────────────
  Obligations:       {z3_total} checked
  Passed:            {z3_pass}/{z3_total} ({z3_pass/z3_total*100:.0f}%)

  FIELD ACCURACY (best field → worst)""")

    sorted_fields = sorted(field_accuracy.items(), key=lambda x: x[1]["correct"]/max(x[1]["total"], 1), reverse=True)
    for field, stats in sorted_fields:
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            print(f"    {field:15s}  {acc:.0%}")

    if all_timings:
        print(f"""
  LATENCY
  ───────
  Avg:               {sum(all_timings)/len(all_timings):.0f}ms
  Total:             {sum(all_timings)/1000:.1f}s""")

    # Results table
    print(f"\n  {'ID':>3s}  {'Name':20s}  {'Extracted':>10s}  {'Actual':>10s}  {'Error':>8s}  {'Route':6s}")
    print(f"  {'─'*3}  {'─'*20}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*6}")
    for e in all_extractions:
        ext_budget = e["extracted"]["budget"]
        gt_budget = e["ground_truth"]["budget"]
        error = abs(ext_budget - gt_budget)
        error_str = f"${error:,}" if error > 0 else "exact"
        print(f"  {e['id']:3d}  {e['ground_truth']['name']:20s}  "
              f"${ext_budget:>9,}  ${gt_budget:>9,}  {error_str:>8s}  "
              f"{'─' * 6}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aura-State Live Benchmark")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--runs", type=int, default=1, help="Extraction runs per transcript (for conformal prediction, use 3+)")
    args = parser.parse_args()
    run(model=args.model, runs_per_transcript=args.runs)
