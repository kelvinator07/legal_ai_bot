"""
MCP stdio server exposing legal tools + web case search.

Spawned by the OpenAI Agents SDK via ``python -m tools.mcp_server`` (cwd = project root).
Run manually for debugging:

    uv run python -m tools.mcp_server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from tools.contract_analyzer import analyze_contract as _analyze_contract
from tools.case_search import find_similar_cases as _find_similar_cases
from tools.complaint_generator import generate_complaint as _generate_complaint
from tools.jurisdiction_detector import detect_jurisdiction as _detect_jurisdiction
from tools.legal_search import search_legal_database as _search_legal_database
from tools.web_cases_search import search_legal_cases_online as _search_legal_cases_online

mcp = FastMCP("legal-ai-bot")


@mcp.tool(name="search_legal_database")
def search_legal_database(topic: str, keyword: str = "", section: str = "") -> dict:
    """Search Nigerian statutes in the local knowledge base by topic and keyword or section."""
    return _search_legal_database(topic=topic, keyword=keyword, section=section)


@mcp.tool(name="find_similar_cases")
def find_similar_cases(description: str, topic: str = "", max_results: int = 5) -> dict:
    """Find relevant statutory provisions for the user's situation from the local knowledge base."""
    return _find_similar_cases(
        description=description, topic=topic, max_results=max_results
    )


@mcp.tool(name="analyze_contract")
def analyze_contract(contract_text: str, country: str = "Nigeria") -> dict:
    """Analyze contract text for risky clauses under Nigerian law."""
    return _analyze_contract(contract_text=contract_text, country=country)


@mcp.tool(name="generate_complaint")
def generate_complaint(
    complaint_type: str,
    user_name: str = "[YOUR NAME]",
    opponent_name: str = "[OPPONENT NAME]",
    facts: str = "",
    topic: str = "",
) -> dict:
    """Generate a complaint email template and filing guide grounded in Nigerian law."""
    return _generate_complaint(
        complaint_type=complaint_type,
        user_name=user_name,
        opponent_name=opponent_name,
        facts=facts,
        topic=topic,
    )


@mcp.tool(name="detect_jurisdiction")
def detect_jurisdiction(user_message: str, stated_country: str = "") -> dict:
    """Detect country / jurisdiction from user message (Pan-African markers)."""
    return _detect_jurisdiction(
        user_message=user_message, stated_country=stated_country
    )


@mcp.tool(name="search_legal_cases_online")
def search_legal_cases_online(query: str, max_results: int = 5) -> dict:
    """Search the web for court decisions, judgments, and legal commentary (verify sources)."""
    return _search_legal_cases_online(query=query, max_results=max_results)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
