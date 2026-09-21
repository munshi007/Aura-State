"""A real LangGraph StateGraph agent — imported by Aura with FULL structure.

    aura-state check examples/code/langgraph_stategraph.py

Unlike a plain @tool scan, Aura parses the StateGraph: the real nodes, the real
edges, AND actions written as graph node functions (send_reply below is an exfil
leg a tool scan never sees). Each node is classified by what its CODE does —
requests.get → untrusted, db.execute → private, smtplib.sendmail → exfil — so the
lethal-trifecta check runs over the actual control flow. Parsed with `ast`; the
code is never imported or run.
"""
import requests
import smtplib
from langgraph.graph import StateGraph, START, END


def fetch_ticket(state):
    """Read the incoming support ticket (attacker-controllable text)."""
    return {"ticket": requests.get(state["url"]).text}


def lookup_account(state):
    """Look up the customer's private account in the database."""
    cur = db.cursor()  # noqa: F821  (illustrative)
    cur.execute("SELECT * FROM customers WHERE id = ?", state["id"])
    return {"account": cur.fetchone()}


def draft_reply(state):
    """LLM drafts a reply from the ticket + account."""
    return {"reply": llm(state)}  # noqa: F821


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
