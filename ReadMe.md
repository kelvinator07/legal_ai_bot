# Legal AI Bot

A **Pan-African legal information assistant** with a production-style layout: a **Gradio** chat UI, an **agent** built with the **OpenAI Agents SDK**, **retrieval-augmented generation (RAG)** over a Nigerian statute corpus, **MCP-exposed tools** for structured lookups, and **web search** for public case law and commentary.

This project provides **legal information only**, not legal advice. Users should consult a qualified lawyer for matters that affect their rights or obligations.

---

## What the codebase does

- **Chat experience:** Users ask questions in a browser UI; replies are produced by an LLM that can call tools over multiple turns.
- **Local knowledge:** Markdown statutes under `rag/knowledge-base/` are chunked, embedded, and stored in a **Chroma** vector database (built when `rag/doc_chunk.py` is loaded). The agent can call **`query_nigerian_statutes_rag`** for semantic retrieval over that index.
- **MCP tool server:** A separate stdio **MCP** process (`tools/mcp_server.py`) exposes tools backed by the same ideas as `tools/*.py`: keyword statute search, scenario-style provision search, contract heuristics, complaint templates, jurisdiction hints, and **online** case-style search via DuckDuckGo.
- **Orchestration:** The agent’s instructions describe when to use RAG vs MCP tools vs web search; the runner connects the MCP server for each request, runs the agent, then tears the connection down (simple lifecycle; you can optimize with a long-lived MCP session later).

---

## Architecture

```
                ┌──────────────────────┐
                │     Gradio UI        │
                └─────────┬────────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │     Agent Layer      │
                │ (LLM + Tool Logic)   │
                └─────────┬────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
 ┌─────────────┐   ┌──────────────┐   ┌──────────────┐
 │     RAG     │   │ MCP Tools    │   │ External APIs│
 │ Vector DB   │   │ (local legal │   │ (web search) │
 │             │   │  workflows)  │   │              │
 └─────────────┘   └──────────────┘   └──────────────┘
```

| Layer | Location | Role |
|--------|-----------|------|
| **UI** | `ui/app.py` | Gradio Blocks, chat history (`role` / `content`), calls the agent runner. |
| **Agent** | `legal_agents/agent.py`, `legal_agents/runner.py` | `Agent` + `Runner.run`, `MCPServerStdio` (`python -m tools.mcp_server`), in-process `function_tool` for RAG. |
| **RAG** | `rag/doc_chunk.py` | Chroma + LangChain retriever + optional second LLM pass to summarize chunks for the agent. |
| **MCP tools** | `tools/mcp_server.py` | FastMCP server: wraps `tools/*.py` implementations. |
| **Web search** | `tools/web_cases_search.py` | DuckDuckGo text search, exposed as MCP tool `search_legal_cases_online`. |

---

## Request flow (end-to-end)

1. The user submits a message in **Gradio** (`ui/app.py`).
2. **`run_legal_agent_sync`** (`legal_agents/runner.py`) builds a single text transcript from prior turns plus the new user line.
3. **`MCPServerManager`** starts the **MCP** subprocess (`tools.mcp_server`) with project root as working directory so `from tools.*` imports resolve.
4. **`Runner.run`** invokes the **LegalAdvisor** agent with the configured **chat-completions**–compatible client (e.g. OpenAI or OpenRouter).
5. The model may call:
   - **`query_nigerian_statutes_rag`** — in-process; runs retrieval + synthesis in `rag/doc_chunk.py`.
   - **MCP tools** — statute search, similar provisions, contract analysis, complaints, jurisdiction, web case search.
6. The final natural-language reply is returned to Gradio and appended to the chat.

---

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or another environment manager
- **GPU optional** — sentence-transformers embeddings run on CPU by default; first load can be slow.

---

## Configuration

Copy `.env.example` to `.env` and set at least:

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | API key for your LLM provider. |
| `OPENAI_BASE_URL` or `OPENROUTER_BASE_URL` | Optional. Defaults to OpenRouter-style base URL in code if unset—override for OpenAI or other compatible endpoints. |
| `LEGAL_AGENT_MODEL` | Optional model id (e.g. `openai/gpt-4o-mini` on OpenRouter). |

The RAG stack in `rag/doc_chunk.py` uses the same OpenAI-compatible client pattern for embeddings/chat as configured there; align keys and base URLs with your provider.

---

## How to run

From the repository root:

```bash
uv sync
cp .env.example .env   # then edit .env
uv run python main.py
```

The app prints a short banner and opens **Gradio** at **http://127.0.0.1:7860** (see `ui/app.py` for host/port).

### Run the MCP server alone (debugging)

```bash
uv run python -m tools.mcp_server
```

Expect stdio MCP traffic on stdin/stdout; useful for MCP clients or troubleshooting imports.

---

## Project layout (high level)

```
legal_ai_bot/
├── main.py                 # Entry: launches Gradio UI
├── ui/app.py               # Gradio interface
├── legal_agents/           # Agent definition and sync runner
├── rag/
│   ├── doc_chunk.py        # Chroma + RAG + rag_query_answer
│   └── knowledge-base/     # Source markdown statutes
├── tools/                  # Tool implementations + mcp_server.py
└── ReadMe.md               # This file
```

---

## Disclaimer

Outputs may be incomplete or wrong. **This is not legal advice.** Laws differ by country and change over time. Verify any citation or filing step with a qualified professional.
