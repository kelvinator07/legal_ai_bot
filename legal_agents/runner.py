"""Sync entrypoint for Gradio: asyncio.run + MCP lifecycle."""

from __future__ import annotations

import asyncio
import logging

from agents import Runner
from agents.mcp import MCPServerManager

from legal_agents.agent import build_legal_mcp_server, create_legal_agent

logger = logging.getLogger(__name__)


def _transcript_for_agent(
    gradio_messages: list | None,
    new_user_message: str,
) -> str:
    """Flatten Gradio chat history + latest user line into one prompt string."""
    lines: list[str] = []
    for msg in gradio_messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content", "")
        text = content if isinstance(content, str) else str(content)
        if role == "user":
            lines.append(f"User: {text}")
        elif role == "assistant":
            lines.append(f"Assistant: {text}")
    lines.append(f"User: {new_user_message.strip()}")
    return "\n".join(lines)


async def run_legal_agent_async(user_message: str, gradio_history: list | None) -> str:
    payload = _transcript_for_agent(gradio_history, user_message)
    mcp = build_legal_mcp_server()
    logger.info("[MCP] MCP server built, entering managed context")
    async with MCPServerManager([mcp], strict=False) as mgr:
        logger.info("[MCP] MCP server connected, %d active servers", len(mgr.active_servers))
        agent = create_legal_agent(mgr.active_servers)
        logger.info("[AGENT] Running agent pipeline (max_turns=25)")
        result = await Runner.run(agent, payload, max_turns=25)
        out = result.final_output
        if isinstance(out, str):
            logger.info("[AGENT] Agent completed, output length=%d chars", len(out))
            return out
        out_str = str(out)
        logger.info("[AGENT] Agent completed, output length=%d chars", len(out_str))
        return out_str


def run_legal_agent_sync(user_message: str, gradio_history: list | None) -> str:
    logger.info("[AGENT] Starting legal agent for user query (length=%d chars)", len(user_message))
    return asyncio.run(run_legal_agent_async(user_message, gradio_history))
