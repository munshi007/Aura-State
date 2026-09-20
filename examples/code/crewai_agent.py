"""A CrewAI-style agent — imported statically by `aura-state check`.

    aura-state check examples/code/crewai_agent.py

The tool classes are mapped to their known roles (web search = untrusted, file
read = private, code interpreter = external execution), so the trifecta check
runs without importing crewai or running anything.
"""
from crewai import Agent
from crewai_tools import SerperDevTool, FileReadTool, CodeInterpreterTool


researcher = Agent(
    role="Researcher",
    goal="Answer questions using the web and local notes",
    tools=[SerperDevTool(), FileReadTool(), CodeInterpreterTool()],
)
