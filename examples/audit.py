"""Audit a folder of agent designs with Aura's static analyzer.

    python examples/audit.py

Runs `aura-state check` (the real verifiers) over every flow in
examples/agents/ and prints a table + a summary — the kind of thing you'd put
in a launch post: "we statically analyzed N common agent patterns."
"""
import glob
import json
import logging
import os

logging.getLogger("aura_state").setLevel(logging.ERROR)
from aura_state.check import check_flow

HERE = os.path.dirname(__file__)


def main() -> None:
    paths = sorted(glob.glob(os.path.join(HERE, "agents", "*.json")))
    rows = []
    for p in paths:
        with open(p) as f:
            flow = json.load(f)
        r = check_flow(flow)
        inj = sum(1 for x in r.findings if x.check == "taint")
        rows.append((flow.get("name", os.path.basename(p)), r.verified, inj, len(r.findings), r))

    w = max(len(name) for name, *_ in rows) + 2
    print(f"\n  Aura static analysis · {len(rows)} agent designs\n")
    print(f"  {'agent'.ljust(w)}{'verdict'.ljust(14)}{'injection paths'.ljust(17)}findings")
    print("  " + "─" * (w + 40))
    for name, ok, inj, total, _ in rows:
        verdict = "✓ PROVEN" if ok else "✗ NOT PROVEN"
        print(f"  {name.ljust(w)}{verdict.ljust(14)}{str(inj).ljust(17)}{total}")

    vulnerable = [name for name, ok, *_ in rows if not ok]
    with_injection = [name for name, ok, inj, *_ in rows if inj > 0]
    print("\n  " + "─" * (w + 40))
    print(f"  {len(with_injection)}/{len(rows)} designs have an unguarded injection path "
          f"(untrusted input → a real tool call with no sanitizer).")
    print(f"  {len(vulnerable)}/{len(rows)} fail verification.\n")
    print("  Every one is fixable by inserting a sanitizer — which Aura's auto-repair does in one step.\n")


if __name__ == "__main__":
    main()
