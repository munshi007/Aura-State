# Cookbook

Real, runnable recipes for using Aura-State on realistic agents. Every script
runs with **no API key** (the LLM boundary is mocked so the guarantees are
reproducible), and each has a one-line switch to run **live against any provider**.

```bash
git clone https://github.com/munshi007/Aura-State && cd Aura-State
pip install -e .
python examples/cookbook/verified_refund_agent.py
```

| Recipe | What it shows |
|---|---|
| [`verified_refund_agent.py`](verified_refund_agent.py) | A high-stakes refund agent with **every** guarantee: Z3 rejects an over-refund in the loop, static taint proves the untrusted message can't reach `issue_refund` without a policy check, CTL proves the escalation path, and it emits an audit contract. |
| [`verify_existing_agent.py`](verify_existing_agent.py) | Add Aura-State's proofs to an agent you **already built** (LangGraph/CrewAI/plain code) — verify its output and its tool graph, no rewrite. |
| [`any_provider.py`](any_provider.py) | The same Aura-State code against OpenAI, Gemini, DeepSeek, Together, or a local model — provider is a one-line change. |

## Use any LLM provider

Aura-State's verification (Z3, CTL, taint, conformal) is independent of the LLM —
only extraction calls a model, and nearly every provider speaks the
OpenAI-compatible API. So switching providers is just a different client:

```python
from openai import OpenAI
import instructor
from aura_state import AuraEngine

# OpenAI
engine = AuraEngine(llm_client=OpenAI())                      # OPENAI_API_KEY

# Gemini (OpenAI-compatible endpoint)
client = OpenAI(api_key=os.environ["GOOGLE_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
engine = AuraEngine(llm_client=instructor.from_openai(client, mode=instructor.Mode.JSON))

# DeepSeek
client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
engine = AuraEngine(llm_client=client)

# Local model via Ollama — no key
client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
engine = AuraEngine(llm_client=instructor.from_openai(client, mode=instructor.Mode.JSON))
```

Then set `Node.model` to that provider's model name (e.g. `"gemini-2.0-flash"`,
`"deepseek-chat"`, `"llama3.1"`). See [`_providers.py`](_providers.py) for a
helper that builds the right client for each.

**Your API key stays yours.** Keys are read from *your* environment (e.g. a local
`.env` outside the repo) — never hard-coded, never committed. When someone else
runs a recipe, it reads *their* key.

## How Aura-State compares to LangGraph / CrewAI / others

See [`docs/COMPARISON.md`](../../docs/COMPARISON.md) — they orchestrate; Aura-State
verifies. Complementary, not competing.
