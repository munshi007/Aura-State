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
