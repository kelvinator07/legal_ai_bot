"""Sync entrypoint for Gradio: asyncio.run + MCP lifecycle."""

from __future__ import annotations

import asyncio

from agents import Runner
from agents.mcp import MCPServerManager

from legal_agents.agent import build_legal_mcp_server, create_legal_agent


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
    async with MCPServerManager([mcp], strict=False) as mgr:
        agent = create_legal_agent(mgr.active_servers)
        result = await Runner.run(agent, payload, max_turns=25)
        out = result.final_output
        if isinstance(out, str):
            return out
        return str(out)


def run_legal_agent_sync(user_message: str, gradio_history: list | None) -> str:
    return asyncio.run(run_legal_agent_async(user_message, gradio_history))
