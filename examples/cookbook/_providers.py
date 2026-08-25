"""
One helper, any provider.

Aura-State's verification (Z3 / CTL / taint / conformal) has nothing to do with
the LLM — only the extraction step calls a model. So "use any provider" just
means handing the engine a client. Almost every provider speaks the
OpenAI-compatible API, so a single `OpenAI(base_url=..., api_key=...)` (patched
by instructor) covers OpenAI, Gemini, DeepSeek, Together, and local models via
Ollama / vLLM / LM Studio.

Keys are read from YOUR environment — never hard-coded, never committed. Set the
one you have (e.g. in a local .env outside the repo) and pass its name below.
"""
import os

import instructor
from openai import OpenAI

# provider -> (env var for the key, base_url or None, default model, instructor mode)
PROVIDERS = {
    "openai":   ("OPENAI_API_KEY",   None,                                                        "gpt-4o-mini",        "TOOLS"),
    "gemini":   ("GOOGLE_API_KEY",   "https://generativelanguage.googleapis.com/v1beta/openai/",  "gemini-2.0-flash",   "JSON"),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com",                                  "deepseek-chat",      "TOOLS"),
    "together": ("TOGETHER_API_KEY", "https://api.together.xyz/v1",                               "meta-llama/Llama-3.3-70B-Instruct-Turbo", "TOOLS"),
    # Local, no key needed — just run `ollama serve` and `ollama pull llama3.1`.
    "ollama":   (None,               "http://localhost:11434/v1",                                 "llama3.1",           "JSON"),
}


def has_key(provider: str) -> bool:
    env, *_ = PROVIDERS[provider]
    return env is None or bool(os.environ.get(env))


def make_client(provider: str = "openai"):
    """Return an instructor-patched client for `provider`, ready for AuraEngine.

    Raises a clear error if the provider's API key isn't set in the environment.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider '{provider}'. Options: {list(PROVIDERS)}")
    env, base_url, _model, mode = PROVIDERS[provider]
    api_key = os.environ.get(env) if env else "not-needed"
    if env and not api_key:
        raise RuntimeError(
            f"{provider}: set {env} in your environment (e.g. a local .env). "
            f"Get a key from the provider's console."
        )
    client = OpenAI(api_key=api_key or "not-needed", base_url=base_url)
    return instructor.from_openai(client, mode=getattr(instructor.Mode, mode))


def default_model(provider: str) -> str:
    return PROVIDERS[provider][2]
