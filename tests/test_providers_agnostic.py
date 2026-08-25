"""Provider-agnostic: any single LLM client serves any model name, and an
already-instructor-patched client is accepted as-is."""
from aura_state.core.providers import LLMProvider


def test_single_client_serves_any_model_name():
    # A non-"gpt" model (gemini/deepseek/local) must route to the sole client.
    p = LLMProvider()
    sentinel = object()
    p.register_client("default", sentinel)
    assert p._get_client_for_model("gemini-2.0-flash") is sentinel
    assert p._get_client_for_model("deepseek-chat") is sentinel
    assert p._get_client_for_model("llama3.1") is sentinel


def test_prefix_still_wins_with_multiple_clients():
    p = LLMProvider()
    gpt, claude = object(), object()
    p.register_client("gpt", gpt)
    p.register_client("claude", claude)
    assert p._get_client_for_model("gpt-4o") is gpt
    assert p._get_client_for_model("claude-3.5-sonnet") is claude
    # No fallback when ambiguous (more than one client, no prefix match).
    assert p._get_client_for_model("gemini-2.0-flash") is None


def test_engine_accepts_prepatched_instructor_client():
    import instructor
    from openai import OpenAI
    from aura_state import AuraEngine

    patched = instructor.from_openai(OpenAI(api_key="x", base_url="http://localhost:1"))
    eng = AuraEngine(llm_client=patched)
    # Not double-patched: the engine used it directly.
    assert eng.client is patched
    # Registered so any model routes to it.
    assert eng.provider._get_client_for_model("gemini-2.0-flash") is patched
