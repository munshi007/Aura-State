"""A bundled, zero-setup demo agent for `aura-state demo`.

The source is embedded (not read from `examples/`) so the demo works from any
clean `pip install aura-state`, no repo, no file paths, no keys. It is a realistic
LangGraph support-ticket agent whose real lethal trifecta Aura catches: an
untrusted web fetch, a private DB read, and an outbound email on one path.
"""
from __future__ import annotations

from typing import Any, Dict

# A real LangGraph StateGraph agent (parsed with `ast`, never executed).
DEMO_SOURCE = '''
import requests, smtplib
from langgraph.graph import StateGraph, START, END

def fetch_ticket(state):
    """Read the incoming support ticket (attacker-controllable text)."""
    return {"ticket": requests.get(state["url"]).text}

def lookup_account(state):
    """Look up the customer's private account in the database."""
    cur = db.cursor()
    cur.execute("SELECT * FROM customers WHERE id = ?", state["id"])
    return {"account": cur.fetchone()}

def draft_reply(state):
    """LLM drafts a reply from the ticket + account."""
    return {"reply": llm(state)}

def send_reply(state):
    """Email the reply to the customer."""
    smtplib.SMTP("mail.internal").sendmail("support@acme.co", state["to"], state["reply"])

g = StateGraph(dict)
g.add_node("fetch", fetch_ticket)
g.add_node("lookup", lookup_account)
g.add_node("draft", draft_reply)
g.add_node("send", send_reply)
g.add_edge(START, "fetch")
g.add_edge("fetch", "lookup")
g.add_edge("lookup", "draft")
g.add_edge("draft", "send")
g.add_edge("send", END)
app = g.compile()
'''


def demo_flow() -> Dict[str, Any]:
    """Build the demo agent's flow via the static code importer (no execution)."""
    from .loaders.code import flow_from_code
    return flow_from_code(DEMO_SOURCE, name="support-ticket-agent")
