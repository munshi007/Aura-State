#!/usr/bin/env python3
"""
Aura-State's Z3 verification over a REAL dataset — at scale, no LLM, no key.

1,000 real sales records (see SOURCE.txt). Each carries arithmetic invariants:
    Total Revenue == Units Sold * Unit Price
    Total Cost    == Units Sold * Unit Cost
    Total Profit  == Total Revenue - Total Cost

Aura-State proves every one with Z3 — the same engine that gates an agent's
extractions. Then it shows the catch: corrupt a single field and Z3 flags it
with a counterexample.

    python examples/real_data/verify_real_dataset.py
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from aura_state import prove_extraction

DATA = os.path.join(os.path.dirname(__file__), "sales_records.csv")
OBLIGATIONS = [
    "total_revenue == units_sold * unit_price",
    "total_cost == units_sold * unit_cost",
    "total_profit == total_revenue - total_cost",
]

def row_record(r):
    return {
        "units_sold":    int(r["Units Sold"]),
        "unit_price":    float(r["Unit Price"]),
        "unit_cost":     float(r["Unit Cost"]),
        "total_revenue": float(r["Total Revenue"]),
        "total_cost":    float(r["Total Cost"]),
        "total_profit":  float(r["Total Profit"]),
    }

def main():
    rows = list(csv.DictReader(open(DATA)))
    print(f"  dataset: {len(rows)} real sales records ({os.path.basename(DATA)})")
    print(f"  invariants per record: {len(OBLIGATIONS)}  ->  {len(rows)*len(OBLIGATIONS)} Z3 obligations\n")

    t0 = time.time()
    verified = violations = 0
    bad_examples = []
    for r in rows:
        rec = row_record(r)
        res = prove_extraction(rec, OBLIGATIONS)
        if res.verified:
            verified += 1
        else:
            violations += 1
            if len(bad_examples) < 3:
                bad_examples.append((r["Order ID"], res.failed_obligations))
    dt = time.time() - t0

    print(f"  VERIFIED: {verified}/{len(rows)} records provably consistent")
    print(f"  violations: {violations}")
    print(f"  throughput: {len(rows)/dt:.0f} records/sec ({len(rows)*len(OBLIGATIONS)/dt:.0f} obligations/sec)")
    for oid, fails in bad_examples:
        print(f"    order {oid} failed: {fails}")

    # ── Show the catch: corrupt one real record's total, watch Z3 flag it ──
    print("\n  Now corrupt one field (total_revenue) to show the catch:")
    rec = row_record(rows[0]); rec["total_revenue"] += 1000
    res = prove_extraction(rec, OBLIGATIONS)
    print(f"    verified: {res.verified}   failed: {res.failed_obligations}")
    print("    -> the same Z3 check that verified 1,000 real records rejects the bad one,")
    print("       with the exact obligation that broke. This is what gates an agent's output.")
    return 0 if violations == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
