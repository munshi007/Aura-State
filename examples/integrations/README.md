# Integrations — real agent frameworks, verified

Use Aura-State as the verification layer over an agent built in a real framework.
Aura-State doesn't replace your orchestrator; it proves things about the agent
it runs.

| Example | What it shows |
|---|---|
| [`langgraph_verified.py`](langgraph_verified.py) | A real **LangGraph** agent running on a **local model (Ollama)** — no API key, nothing sent to the cloud. Aura-State proves the agent's extracted output with Z3 and proves its tool graph is injection-safe, then emits an audit contract. |

## Run it live (real model, no key, no cloud)

```bash
# 1. a local model via Ollama
brew install ollama            # or your platform's installer
ollama serve &                 # start the local server
ollama pull llama3.2:1b        # ~1.3 GB, one time (small model is plenty)

# 2. the framework + this repo
pip install langgraph
pip install -e .

python examples/integrations/langgraph_verified.py
```

With no Ollama running it still executes end-to-end using a deterministic stub
for the LLM node (clearly labeled), so you can see the LangGraph + Aura-State
wiring without any setup.
