#!/usr/bin/env python3
"""
Same Aura-State code, any LLM provider.

Aura-State's verification doesn't care which model you use — only the extraction
step calls one, and almost every provider speaks the OpenAI-compatible API. So
"switch providers" is a one-line change: point the client at a different URL.

    python examples/cookbook/any_provider.py                 # shows the recipe for all providers
    python examples/cookbook/any_provider.py --provider gemini   # live extraction (needs GOOGLE_API_KEY)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

from pydantic import BaseModel, Field
from aura_state import AuraEngine, Node, CompiledTransition
from _providers import PROVIDERS, make_client, default_model, has_key


class Lead(BaseModel):
    name: str = Field(description="the person's name")
    budget: int = Field(description="budget in whole USD")


class Extract(Node):
    system_prompt = "Extract the lead's name and budget."
    extracts = Lead
    obligations = ["budget >= 0"]     # proven in the loop, same for every provider

    def handle(self, user_text, extracted_data=None, memory=None):
        return "END", extracted_data.model_dump() if extracted_data else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()

    print("Point Aura-State at any provider — the code below is identical for all:\n")
    for name, (env, base_url, model, mode) in PROVIDERS.items():
        key = "no key needed" if env is None else (f"{env} " + ("✓ set" if has_key(name) else "✗ not set"))
        print(f"  {name:9} model={model:38} base_url={base_url or 'default'}")
        print(f"  {'':9} key: {key}")
    print("\n  engine = AuraEngine(llm_client=make_client('<provider>'))")
    print("  # then everything — Z3, CTL, taint, conformal — works the same.\n")

    if not args.provider:
        print("Pass --provider <name> to run a live extraction with a key you have.")
        return 0

    if not has_key(args.provider):
        env = PROVIDERS[args.provider][0]
        print(f"Set {env} in your environment to run {args.provider} live.")
        return 1

    print(f"Live extraction via {args.provider} ({default_model(args.provider)}):")
    engine = AuraEngine(llm_client=make_client(args.provider))
    for n in engine._nodes.values():
        pass
    engine.register(Extract)
    engine._nodes["Extract"].model = default_model(args.provider)
    engine._transitions["Extract"] = ["END"]
    engine.process("Extract", "Hi, I'm Sarah and my budget is 450000 dollars.")
    rep = engine.verification_reports()[-1]
    print(f"  extracted + verified in the loop: {rep['extraction_verified']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
