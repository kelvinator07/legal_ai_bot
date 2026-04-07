"""Legal advisor agent: MCP tools (statutes, web) + in-process RAG tool."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from agents import Agent, ModelSettings, function_tool, set_default_openai_api
from agents.mcp import MCPServerStdio
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from rag.doc_chunk import rag_query_answer


load_dotenv(override=True)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

set_default_openai_api("chat_completions")


def _make_openai_client() -> AsyncOpenAI:
    key = os.getenv("OPENAI_API_KEY")
    logger.info("[AGENT] Initializing OpenAI client")
    logger.debug("[AGENT] API key present: %s", bool(key))
    return AsyncOpenAI(api_key=key)


def _model_name() -> str:
    name = os.getenv("LEGAL_AGENT_MODEL", "gpt-4.1-mini")
    logger.info("[AGENT] Model: %s", name)
    return name


@function_tool
def query_nigerian_statutes_rag(user_question: str) -> str:
    """Semantic search over the indexed Nigerian law corpus (vector DB).

    Use for questions that need consolidated context from multiple sections or
    when MCP keyword search is too narrow. Does not replace statute lookup tools
    for precise section numbers.
    """
    logger.info("[RAG] query_nigerian_statutes_rag invoked (question length=%d)", len(user_question))
    return rag_query_answer(user_question, None)


def build_legal_mcp_server() -> MCPServerStdio:
    logger.info("[MCP] Building MCPServerStdio (command=%s)", sys.executable)
    # Note: Use -m as args so project root is on sys.path (running mcp_server.py as a file breaks `from tools.*`).
    return MCPServerStdio(
        params={
            "command": sys.executable,
            "args": ["-m", "tools.mcp_server"],
            "cwd": str(PROJECT_ROOT),
        },
        cache_tools_list=True,
        name="legal-mcp",
        client_session_timeout_seconds=60.0,
    )


def create_legal_agent(mcp_servers: list) -> Agent:
    logger.info("[AGENT] Creating LegalAdvisor agent with %d MCP servers", len(mcp_servers))
    client = _make_openai_client()
    model = OpenAIChatCompletionsModel(model=_model_name(), openai_client=client)

    instructions = """You are a Pan-African professional legal information assistant focused on Nigerian law,
    tasked to answer question related to human right violation, criminal case, site examples,
provides constitutional act related to the user's question when the user's jurisdiction is Nigeria,
and provides general guidance for other African jurisdictions.

Use tools deliberately:
- search_legal_database: keyword/topic search in local statutes.
- find_similar_cases: map a situation to relevant statutory provisions in the local KB.
- analyze_contract: when the user pastes contract text.
- generate_complaint: when the user wants a filing or complaint draft.
- detect_jurisdiction: early in the conversation if country is unclear.
- search_legal_cases_online: web search for reported cases, judgments, or commentary \
  (always warn that web hits must be verified).
- query_nigerian_statutes_rag: semantic retrieval over the vector index for broad questions.

Rules:
1. Prefer local knowledge-base tools before web search for what Nigerian statutes say.
2. Never present web search results as authoritative holdings; cite sources and urge verification.
3. End substantive answers with a short disclaimer that this is legal information for general guidance and educational purpose, visit legal practitioner for more detailed legal advice.
4. Professionally ignore unrelated question, focus on legal and constitutional issues.
"""

    return Agent(
        name="LegalAdvisor",
        instructions=instructions,
        model=model,
        model_settings=ModelSettings(temperature=0.3),
        tools=[query_nigerian_statutes_rag],
        mcp_servers=mcp_servers,
    )
