# Customer Support Copilot — Generative AI Multi-Agent System

A multi-agent assistant that lets a support executive (John) ask natural-language
questions and get context-aware answers pulled from **both**:
- a **structured SQL database** of customers and support tickets, and
- an **unstructured knowledge base** of company policy PDFs (searched via a vector store).

Built with **LangGraph** (multi-agent orchestration), a standalone **MCP server**
(exposes the data tools), **LangChain**, **Chroma** (vector DB), **SQLite** (structured DB),
and a **Streamlit** chat UI.

## Demo video

`<PASTE YOUR DEMO VIDEO URL HERE>`

## Architecture

```
┌─────────────────────┐        ┌──────────────────────────────────────────┐
│   Streamlit UI       │        │              MCP Server                   │
│  (app/streamlit_app) │        │           (mcp_server/server.py)          │
│                      │  MCP   │  Tools:                                   │
│  - chat interface     │ over  │   - get_database_schema                   │
│  - PDF upload widget  │ HTTP  │   - run_sql_query        ──► SQLite       │
└─────────┬────────────┘        │   - get_customer_profile     (customers,  │
          │                     │   - list_customers            support_    │
          │ calls               │   - search_policy_documents    tickets)   │
          ▼                     │   - reingest_policy_documents ──► Chroma  │
┌─────────────────────────┐     │                                (policy    │
│   LangGraph Multi-Agent  │◄────┘                                 PDFs)    │
│      (agents/graph.py)   │                                                │
│                          │     └──────────────────────────────────────────┘
│   router (LLM classify) │
│     ├─► sql_agent  ─────┼── ReAct agent bound to the 4 SQL/customer tools
│     ├─► docs_agent ─────┼── ReAct agent bound to the policy-search tool
│     └─► synthesizer     │── combines findings into one final answer
└─────────────────────────┘
```

**Why this shape is a "multi-agent system"** and not a single chatbot with tools:
a lightweight **router** node classifies every incoming message (does it need
customer/ticket data? does it need policy-document knowledge? both? neither?), then
fans out to independently-scoped specialist agents — a **SQL agent** that can only see
database tools, and a **document agent** that can only see the vector-search tool — and a
final **synthesizer** merges whichever findings came back into one clear, cited answer.
Both specialist agents get their tools exclusively through the **MCP server**, not by
calling Python functions directly, so the tool layer is a real, independently-runnable
service any MCP-compatible client could reuse.

### Component breakdown

| Component | File(s) | Purpose |
|---|---|---|
| Structured data | `scripts/seed_database.py` → `data/support.db` | Synthetic SQLite DB: `customers` + `support_tickets` tables (Faker-generated, includes a featured customer "Ema Thompson" with a full ticket history, matching the assignment's example query). |
| Unstructured data | `scripts/generate_policy_pdfs.py` → `data/policies/*.pdf` | 4 dummy company policy PDFs: Refund, Privacy, Shipping, Support SLA. |
| Ingestion pipeline | `scripts/ingest_documents.py`, `mcp_server/knowledge_base.py` | Loads PDFs → chunks (`RecursiveCharacterTextSplitter`) → embeds locally (`sentence-transformers/all-MiniLM-L6-v2`, no API key needed) → persists to a **Chroma** vector store at `data/chroma_db/`. |
| MCP server | `mcp_server/server.py` | A `FastMCP` server exposing 6 tools (SQL schema/query/customer-lookup + policy vector search/re-ingest) over streamable-HTTP. This is the single access point for all data — both structured and unstructured. |
| Multi-agent orchestrator | `agents/graph.py`, `agents/config.py` | A `LangGraph` `StateGraph`: `router → {sql_agent, docs_agent} → synthesizer`. Connects to the MCP server as a client via `langchain-mcp-adapters`. |
| UI | `app/streamlit_app.py` | Chat interface + sidebar PDF uploader (ingests new policy docs at runtime) + MCP connection status. |

### Why an MCP server (instead of calling the DB/vector-store directly from the agent)?

Putting the actual data access behind an MCP server means the tool surface
(`run_sql_query`, `search_policy_documents`, etc.) is a standalone, protocol-compliant
service — it could be reused by Claude Desktop, another agent framework, or a second UI
without any code changes. The LangGraph agent is just one MCP *client* among possibly many.

## Tech stack

- **Orchestration**: LangGraph, LangChain
- **LLM**: configurable — defaults to a **free local model via Ollama** (`llama3.1:8b`, no
  API key required). OpenAI and Anthropic are also supported via `.env`.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local, free, no API key)
- **Structured DB**: SQLite (`data/support.db`)
- **Vector DB**: Chroma (`data/chroma_db/`)
- **Tool server**: MCP (`mcp` Python SDK, `FastMCP`, streamable-HTTP transport)
- **UI**: Streamlit

## Setup

### 1. Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/download) installed, **or** an OpenAI/Anthropic API key

### 2. Install

```bash
git clone <this-repo-url>
cd customer-support-multiagent
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### 3. Choose your LLM provider (edit `.env`)

**Option A — free, local (default):**
```bash
brew install ollama              # or see https://ollama.com/download
brew services start ollama       # or: ollama serve
ollama pull llama3.1:8b
```
`.env` already defaults to `LLM_PROVIDER=ollama`.

**Option B — OpenAI:** set `LLM_PROVIDER=openai` and `OPENAI_API_KEY=sk-...` in `.env`.

**Option C — Anthropic:** set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY=...` in `.env`.

(Embeddings always run locally — no key needed for the vector store regardless of provider.)

### 4. Seed the data

```bash
python scripts/seed_database.py       # creates data/support.db (40 customers, ~100 tickets)
python scripts/generate_policy_pdfs.py  # creates data/policies/*.pdf
python scripts/ingest_documents.py    # embeds PDFs into data/chroma_db/
```

### 5. Run

Two processes, in separate terminals (both need the venv activated):

```bash
# Terminal 1 — the MCP tool server
python -m mcp_server.server

# Terminal 2 — the Streamlit UI
streamlit run app/streamlit_app.py
```

Open the URL Streamlit prints (typically http://localhost:8501).

## Usage

- Ask a policy question, e.g. **"What is the current refund policy?"** — the router
  detects this needs the knowledge base, the docs agent searches the vector store, and
  the synthesizer returns a cited summary.
- Ask about a customer, e.g. **"Give me a quick overview of customer Ema's profile and
  past support ticket details."** — the router detects this needs structured data, the
  SQL agent looks up the customer + their tickets, and the synthesizer returns a summary.
- Ask something that needs both, e.g. **"Ema had a refund issue — what does our refund
  policy actually say about that?"** — both agents run and the synthesizer combines them.
- Upload a new policy PDF from the sidebar — it's chunked, embedded, and immediately
  searchable in the next question, no restart required.

## Project structure

```
customer-support-multiagent/
├── agents/
│   ├── config.py        # env config, LLM/embeddings factory
│   └── graph.py          # LangGraph multi-agent graph (router/sql_agent/docs_agent/synthesizer)
├── mcp_server/
│   ├── server.py          # FastMCP server exposing the 6 data tools
│   └── knowledge_base.py  # Chroma vector store access + PDF ingestion
├── scripts/
│   ├── seed_database.py       # generates synthetic customers + tickets → SQLite
│   ├── generate_policy_pdfs.py # generates the 4 dummy policy PDFs
│   └── ingest_documents.py     # CLI to (re)ingest data/policies/*.pdf
├── app/
│   └── streamlit_app.py  # chat UI + PDF uploader
├── data/
│   ├── support.db         # SQLite DB (generated)
│   ├── policies/*.pdf     # policy PDFs (generated)
│   └── chroma_db/         # vector store (generated)
├── requirements.txt
└── .env.example
```

## Notes / design decisions

- **SQL safety**: `run_sql_query` only accepts `SELECT` statements (rejects
  `INSERT`/`UPDATE`/`DELETE`/`DROP`/etc. and multi-statement input) and connects to
  SQLite in read-only mode (`mode=ro`), since the agent is generating SQL from natural
  language and should never be able to mutate data.
- **Local-first by default**: both the embedding model and the default LLM (Ollama) run
  entirely on-device, so the whole system works with zero API keys and zero cost — useful
  for grading/demo purposes without requiring reviewers to provision credentials.
- **Synthetic data**: customer/ticket data is Faker-generated (seeded, reproducible) and
  policy PDFs are original dummy documents written for this project (per the assignment's
  "any synthetic or dummy...any publicly available" data guidance).
