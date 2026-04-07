# Legal AI Bot

A **Pan-African legal information assistant** with a focus on Nigerian law: a **Gradio** chat UI, an **agent** built with the **OpenAI Agents SDK**, **retrieval-augmented generation (RAG)** over a Nigerian statute corpus, **MCP-exposed tools** for structured lookups, and **web search** for public case law and commentary.

This project provides **legal information only**, not legal advice. Users should consult a qualified lawyer for matters that affect their rights or obligations.

---

## What the codebase does

- **Chat experience:** Users ask questions in a browser UI; replies are produced by an LLM that can call tools over multiple turns.
- **Local knowledge:** Markdown statutes under `rag/knowledge-base/` are chunked, embedded, and stored in a **Chroma** vector database (built when `rag/doc_chunk.py` is loaded). The agent can call **`query_nigerian_statutes_rag`** for semantic retrieval over that index.
- **MCP tool server:** A separate stdio **MCP** process (`tools/mcp_server.py`) exposes six tools backed by `tools/*.py`: keyword statute search, scenario-style provision search, contract analysis, complaint templates, jurisdiction detection, and **online** case search via DuckDuckGo.
- **Orchestration:** The agent's instructions describe when to use RAG vs MCP tools vs web search; the runner connects the MCP server for each request, runs the agent, then tears the connection down.

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
| **RAG** | `rag/doc_chunk.py` | Chroma + LangChain retriever + second LLM pass to summarize chunks for the agent. |
| **MCP tools** | `tools/mcp_server.py` | FastMCP server: wraps `tools/*.py` implementations. |
| **Web search** | `tools/web_cases_search.py` | DuckDuckGo text search, exposed as MCP tool `search_legal_cases_online`. |

---

## Tools

The agent has access to **7 tools** (6 MCP + 1 in-process):

| Tool | Source | Description |
|------|--------|-------------|
| `search_legal_database` | `tools/legal_search.py` | Keyword/topic search across local statutes (labor, tenancy, consumer, constitution, food). |
| `find_similar_cases` | `tools/case_search.py` | Maps a user's situation description to relevant statutory provisions via keyword-to-scenario matching. |
| `analyze_contract` | `tools/contract_analyzer.py` | Flags risky clauses in contract text (e.g. liability waivers, no-refund terms) with legal citations. |
| `generate_complaint` | `tools/complaint_generator.py` | Generates a formal complaint letter with filing instructions, required documents, and timelines. |
| `detect_jurisdiction` | `tools/jurisdiction_detector.py` | Identifies the user's country from currency, city names, slang, and legal terminology. |
| `search_legal_cases_online` | `tools/web_cases_search.py` | Web search for reported cases and legal commentary via DuckDuckGo. |
| `query_nigerian_statutes_rag` | `legal_agents/agent.py` | In-process semantic retrieval over the Chroma vector index for broad legal questions. |

---

## Knowledge base

The RAG pipeline indexes **7 Nigerian legal documents** stored as Markdown in `rag/knowledge-base/`:

| Document | Coverage |
|----------|----------|
| Nigeria Constitution 1999 | Fundamental rights, citizenship, governance |
| Labour Act | Wages, contracts, termination, leave, overtime |
| Federal Consumer Act (FCCPA 2018) | Consumer rights, refunds, defective goods |
| Consumer Act | General consumer protection |
| Tenancy Law (Lagos 2011) | Tenant rights, eviction, notice periods |
| Tenancy Disputes | Property dispute resolution analysis |
| Food And Drugs Act | Food/drug manufacturing and sale |

---

## Guardrails

### Input

| Layer | Measure | Limit |
|-------|---------|-------|
| UI | Message length | 2,000 characters |
| UI | Null byte removal | Sanitized before processing |
| UI | Empty message | Rejected |
| RAG | Question length | 3,000 characters (capped) |
| Agent | Max turns | 25 |
| MCP | Server timeout | 60 seconds |

### Output

| Layer | Measure | Limit |
|-------|---------|-------|
| RAG | Final answer length | 4,000 characters (truncated with "...") |
| Agent | Disclaimer | Appended to substantive answers |
| UI | Error handling | User-friendly error messages |
| Web search | Disclaimer | Results flagged as unverified |

---

## Request flow (end-to-end)

1. The user submits a message in **Gradio** (`ui/app.py`).
2. **`run_legal_agent_sync`** (`legal_agents/runner.py`) builds a single text transcript from prior turns plus the new user line.
3. **`MCPServerManager`** starts the **MCP** subprocess (`tools.mcp_server`) with project root as working directory so `from tools.*` imports resolve.
4. **`Runner.run`** invokes the **LegalAdvisor** agent with the configured **chat-completions**-compatible client.
5. The model may call:
   - **`query_nigerian_statutes_rag`** -- in-process; runs retrieval + synthesis in `rag/doc_chunk.py`.
   - **MCP tools** -- statute search, similar provisions, contract analysis, complaints, jurisdiction, web case search.
6. The final natural-language reply is returned to Gradio and appended to the chat.

---

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or another environment manager
- **GPU optional** -- sentence-transformers embeddings run on CPU by default; first load can be slow.

---

## Configuration

Copy `.env.example` to `.env` and set at least:

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENAI_API_KEY` | API key for OpenAI. | *(required)* |
| `LEGAL_AGENT_MODEL` | Model identifier for the agent LLM. | `gpt-4.1-mini` |

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

## Project layout

```
legal_ai_bot/
├── main.py                       # Entry: launches Gradio UI
├── pyproject.toml                # Dependencies and project metadata
├── .env.example                  # Environment variable template
├── ui/
│   └── app.py                    # Gradio chat interface
├── legal_agents/
│   ├── agent.py                  # Agent definition, RAG tool, MCP server builder
│   └── runner.py                 # Sync runner (asyncio.run + MCP lifecycle)
├── rag/
│   ├── doc_chunk.py              # Chroma vector DB setup, retrieval, synthesis
│   ├── knowledge-base/           # 7 Nigerian statute markdown files
│   └── chat_vector_db/           # Persisted Chroma vector store
├── tools/
│   ├── mcp_server.py             # FastMCP stdio server (spawned as subprocess)
│   ├── legal_search.py           # search_legal_database
│   ├── case_search.py            # find_similar_cases
│   ├── contract_analyzer.py      # analyze_contract
│   ├── complaint_generator.py    # generate_complaint
│   ├── jurisdiction_detector.py  # detect_jurisdiction
│   ├── web_cases_search.py       # search_legal_cases_online
│   ├── utils.py                  # Shared utilities (knowledge base loader)
│   └── TOOLS_DOCUMENTATION.md    # Tool reference documentation
└── ReadMe.md                     # This file
```

---

## Disclaimer

Outputs may be incomplete or wrong. **This is not legal advice.** Laws differ by country and change over time. Verify any citation or filing step with a qualified professional.
