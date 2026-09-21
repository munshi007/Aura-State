"""Tests for the source-code importer (LangGraph / CrewAI / LangChain -> flow)."""
from aura_state.loaders.code import flow_from_code, _extract_tools
from aura_state.check import check_flow


LANGGRAPH_SRC = '''
from langchain_core.tools import tool

@tool
def fetch_url(url: str) -> str:
    "Fetch a URL and return its contents."
    return do_fetch(url)          # undefined on purpose: we must NOT execute this

@tool("read_customer_file")
def _reader(path: str) -> str:
    "Read a customer record from a local file."
    ...

@tool
def post_to_slack(text: str) -> str:
    "Post a message to a Slack channel."
    ...

agent = create_react_agent(model, tools=[fetch_url, _reader, post_to_slack])
'''

CREWAI_SRC = '''
from crewai import Agent
from crewai_tools import SerperDevTool, FileReadTool, CodeInterpreterTool
a = Agent(tools=[SerperDevTool(), FileReadTool(), CodeInterpreterTool()])
'''


def test_extracts_tool_decorated_functions_without_executing():
    tools = {t["name"] for t in _extract_tools(LANGGRAPH_SRC)}
    # note the decorator-name override: @tool("read_customer_file") wins over _reader
    assert {"fetch_url", "read_customer_file", "post_to_slack"} <= tools


def test_langgraph_agent_closes_trifecta():
    r = check_flow(flow_from_code(LANGGRAPH_SRC, "lg"))
    assert r.verified is False
    tri = [f for f in r.findings if f.check == "trifecta" and f.severity == "critical"]
    assert tri and tri[0].node == "post_to_slack"


def test_crewai_known_tools_mapped_to_roles():
    r = check_flow(flow_from_code(CREWAI_SRC, "crew"))
    assert r.verified is False
    tri = [f for f in r.findings if f.check == "trifecta" and f.severity == "critical"]
    # SerperDevTool (untrusted) -> CodeInterpreterTool (exfil) with FileReadTool (private)
    assert any(f.node == "CodeInterpreterTool" for f in tri)


def test_tool_and_structuredtool_constructors():
    src = ('t = Tool(name="send_email", description="Send an email to a user")\n'
           's = StructuredTool.from_function(name="lookup_account", description="Read a CRM account")\n')
    names = {t["name"] for t in _extract_tools(src)}
    assert {"send_email", "lookup_account"} <= names


def test_import_never_executes_module_side_effects():
    # references to undefined names / imports that would crash on exec must be fine
    src = 'from nope import missing\n@tool\ndef f(x):\n    "read a file"\n    return missing()\n'
    # tool decorator not imported as `tool` here, but the decorator name is `tool`
    tools = _extract_tools("from langchain.tools import tool\n" + src)
    assert any(t["name"] == "f" for t in tools)   # parsed, not run


LANGGRAPH_STATEGRAPH = '''
import requests, smtplib
from langgraph.graph import StateGraph, START, END
def fetch_ticket(state):
    "Read the incoming support ticket"
    return {"t": requests.get(state["url"]).text}
def lookup_account(state):
    "Look up the private customer account"
    cur = db.cursor(); cur.execute("select * from customers"); return {"a": cur.fetchone()}
def draft(state):
    return {"r": llm(state)}
def send_reply(state):
    "Email the reply to the customer"
    smtplib.SMTP("m").sendmail("f", state["to"], state["r"])
g = StateGraph(dict)
g.add_node("fetch", fetch_ticket)
g.add_node("lookup", lookup_account)
g.add_node("draft", draft)
g.add_node("send", send_reply)
g.add_edge(START, "fetch")
g.add_edge("fetch", "lookup")
g.add_edge("lookup", "draft")
g.add_edge("draft", "send")
g.add_edge("send", END)
app = g.compile()
'''


def test_langgraph_stategraph_is_captured_with_real_structure():
    from aura_state.loaders.code import flow_from_code
    flow = flow_from_code(LANGGRAPH_STATEGRAPH, "support")
    ids = {n["id"] for n in flow["nodes"]}
    # the graph node functions are captured, incl. the exfil node a @tool scan misses
    assert {"fetch", "lookup", "draft", "send"} <= ids
    assert ["fetch", "lookup"] in flow["edges"] and ["draft", "send"] in flow["edges"]
    assert flow["entry"] == "fetch"


def test_langgraph_node_classified_by_what_its_code_does():
    from aura_state.loaders.code import flow_from_code
    flow = flow_from_code(LANGGRAPH_STATEGRAPH, "support")
    roles = {n["id"]: set(n.get("roles", [])) for n in flow["nodes"]}
    assert "untrusted" in roles["fetch"]        # requests.get
    assert "private" in roles["lookup"]         # db.cursor/execute
    assert "exfil" in roles["send"]             # smtplib.sendmail


def test_langgraph_trifecta_over_real_flow():
    r = check_flow(flow_from_code(LANGGRAPH_STATEGRAPH, "support"))
    assert r.verified is False
    tri = [f for f in r.findings if f.check == "trifecta" and f.severity == "critical"]
    # the exfil sink is the send node (which a @tool scan never saw); the key is
    # "<untrusted-source>-><exfil-sink>" on the real path
    assert tri and any(f.node == "send" and f.key.endswith("->send") for f in tri)


CREW = '''
from crewai import Agent, Crew, Task
from crewai_tools import SerperDevTool, FileReadTool, CodeInterpreterTool, ScrapeWebsiteTool
researcher = Agent(role="researcher", goal="g", tools=[SerperDevTool(), ScrapeWebsiteTool()])
analyst = Agent(role="analyst", goal="g", tools=[FileReadTool(), CodeInterpreterTool()])
crew = Crew(agents=[researcher, analyst], tasks=[Task(description="x", agent=researcher)])
'''


def test_crewai_tools_are_scoped_per_agent():
    from aura_state.loaders.code import flow_from_code
    flow = flow_from_code(CREW, "crew")
    ids = {n["id"] for n in flow["nodes"]}
    # each agent's tools are namespaced to that agent, not merged into one hub
    assert "researcher:SerperDevTool" in ids and "analyst:CodeInterpreterTool" in ids
    assert ["researcher", "analyst"] in flow["edges"]   # sequential hand-off


def test_crewai_trifecta_only_via_real_handoff():
    r = check_flow(flow_from_code(CREW, "crew"))
    tri = [f for f in r.findings if f.check == "trifecta" and f.severity == "critical"]
    # untrusted (researcher scrape) -> exfil (analyst code interp), via the R->A hand-off
    assert tri and any("analyst:" in (f.node or "") for f in tri)
