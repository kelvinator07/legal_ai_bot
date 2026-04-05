"""
Legal AI Bot - Tool Use Module
==============================
Pure-Python tool implementations used by the MCP server (`tools.mcp_server`) and
direct imports. The production agent loads these via **MCP stdio** (see `legal_agents`).

Tools (also exposed as MCP tools):
    - search_legal_database
    - analyze_contract
    - find_similar_cases
    - generate_complaint
    - detect_jurisdiction
    - search_legal_cases_online (web; see `web_cases_search`)

Run MCP server locally:
    uv run python -m tools.mcp_server
"""

from tools.legal_search import search_legal_database
from tools.contract_analyzer import analyze_contract
from tools.case_search import find_similar_cases
from tools.complaint_generator import generate_complaint
from tools.jurisdiction_detector import detect_jurisdiction

# Ready-made list your teammates can pass straight to an Agent
ALL_TOOLS = [
    search_legal_database,
    analyze_contract,
    find_similar_cases,
    generate_complaint,
    detect_jurisdiction,
]
