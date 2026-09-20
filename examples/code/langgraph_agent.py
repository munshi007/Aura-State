"""A LangGraph-style ReAct agent — the shape `aura-state check` imports statically.

    aura-state check examples/code/langgraph_agent.py

Aura parses this file; it does NOT run it (no LLM, no network). It extracts the
@tool functions, models the worst case (the agent may call any tool in any
order), and checks whether they close the lethal trifecta.
"""
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


@tool
def fetch_url(url: str) -> str:
    """Fetch a URL and return its page contents as text."""
    ...


@tool
def read_customer_file(path: str) -> str:
    """Read a customer record from a local file on disk."""
    ...


@tool
def post_to_slack(channel: str, text: str) -> str:
    """Post a message to a Slack channel."""
    ...


agent = create_react_agent(
    "openai:gpt-4o",
    tools=[fetch_url, read_customer_file, post_to_slack],
)
