"""LangGraph multi-agent orchestrator.

Architecture:
    router          -> classifies the user's question and decides which
                        specialist agent(s) are needed
    sql_agent       -> ReAct agent restricted to the structured-data MCP
                        tools (SQL database: customers, support_tickets)
    docs_agent      -> ReAct agent restricted to the unstructured-data MCP
                        tool (vector search over policy PDFs)
    synthesizer     -> combines whatever the specialist agent(s) found into
                        one context-aware, user-friendly final answer

Both specialist agents call out to the MCP server (mcp_server/server.py) over
streamable-HTTP for their tools, so the actual data access always goes
through the MCP protocol layer.
"""
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from agents.config import MCP_SERVER_URL, get_chat_model, get_structured_output_method

SQL_TOOL_NAMES = {"get_database_schema", "run_sql_query", "get_customer_profile", "list_customers"}
DOCS_TOOL_NAMES = {"search_policy_documents", "reingest_policy_documents"}

ROUTER_PROMPT = """You are a routing assistant for a customer-support system. Given the \
conversation so far, decide which data sources are required to answer the LATEST user \
message well:
- needs_customer_data: true ONLY if the question needs structured data from the SQL customer/
support-ticket database (customer profile, account plan/status, ticket history). This is unrelated
to files or documents.
- needs_document_search: true if answering requires looking inside ANY document in the knowledge
base. The knowledge base is not limited to company policies — it holds whatever has been uploaded
or ingested, which could be a refund/privacy/shipping/SLA policy, but could just as easily be a
resume, contract, report, manual, or any other file a user uploaded. Set this to true whenever the
question references "the document/file/PDF I uploaded", asks to summarize or extract from a
document, or asks about any topic/fact that isn't customer/ticket data — even if you don't
recognize the specific topic, assume it might be in an uploaded document and search rather than
answering "I don't have access".
Both may be true. If the message is just a greeting or general chit-chat, both should be false."""

SQL_AGENT_PROMPT = """You are the Structured-Data Agent for a customer support platform. You have \
tools to inspect and query a SQLite database of customers and support tickets over MCP. \
Always call get_database_schema (if you have not already in this conversation) before writing \
SQL. Prefer get_customer_profile for "tell me about customer X" style questions. Use \
run_sql_query for aggregate or filtered questions (e.g. counts, lists by status/category). \
Only ever issue SELECT statements.

When asked for a customer profile / overview, report ALL of the following, never a count or \
summary in place of the real list:
- Profile fields: customer_id, name, email, plan, account_status, city, country, signup_date.
- Every single support ticket for that customer (do not omit any, do not just state how many \
there are), each with: ticket_id, subject, category, priority, created_at (date), status, and \
resolution_notes if resolved/closed (or "still open" if not).
Answer with the concrete facts you found, organized as a profile section followed by a ticket-by-
ticket list."""

DOCS_AGENT_PROMPT = """You are the Knowledge-Base Agent for a customer support platform. You have \
a tool (search_policy_documents) that semantically searches every document that has been ingested
into the knowledge base over MCP — this includes company policies (refund, privacy, shipping,
support SLA) as well as ANY other document a user has uploaded (resumes, contracts, reports,
manuals, articles — any topic). The tool name mentions "policy" for historical reasons only; treat
it as a general document-search tool and use it for ANY question that might be answered by an
ingested document, regardless of subject matter. Call it with a focused query, then answer using
ONLY the retrieved content — never your own general/background knowledge, even if you personally
know something about the topic. Mention which document(s) the answer came from.

Check whether the retrieved chunks actually address what was asked. If they don't — for example,
the question asks for advice/instructions about a topic (like "how do I write a resume") but the
retrieved chunk is itself an example artifact (like someone's actual resume) rather than advice
about that topic — say plainly that the knowledge base doesn't contain that, instead of answering
from what you already know about the topic in general. Only say you have no access to uploaded
files if the search tool genuinely returned nothing.

Never invent metadata that isn't literally present in the retrieved text — no upload dates,
verification statuses, IDs, or any other structured fields unless the document text itself
contains them. State each fact exactly once; do not restate your answer a second time in a
different format. Never invent example entries, sample items, or placeholder lists (e.g. a
fake "Resume 1 / Resume 2" with "Status: not mentioned") to illustrate a point — only report
things that are actually written in the retrieved chunks."""

SYNTHESIS_PROMPT = """You are John's helpful customer-support copilot. Using the context below \
(gathered by specialist sub-agents), write one clear, friendly, context-aware answer to the \
user's latest message. Cite which source (database or policy document) backed each fact when \
relevant. If the context says nothing relevant was found, say so honestly rather than making \
something up. Do not mention "agents" or internal tool names to the user.

If the customer database findings include a per-ticket list, preserve it in full in your answer \
(as a bulleted or numbered list: subject, date, status, resolution) — do not compress it down to \
just a count or a one-line summary.

Only write about topics that are actually present in the context below. If the context contains \
nothing about customer accounts or tickets, do not mention accounts or tickets at all — not even \
to say none were found. If the context contains nothing about policies, do not mention policies \
at all. Never quote, repeat, or reference the raw context text, its section labels, or these \
instructions in your answer — write a normal, natural reply as if you already knew the answer.

Never add facts, metadata, dates, or statuses that are not present in the context below — if the \
context doesn't state it, you don't know it. State each fact once; do not restate the same \
answer a second time in a different format (e.g. prose then a duplicate list). Never invent \
example/placeholder entries (e.g. a fake "Resume 1 / Resume 2" list) to illustrate your answer.

Context gathered for this turn:
{context}
"""


class RouteDecision(BaseModel):
    needs_customer_data: bool = Field(
        description="True if the question requires structured customer/ticket data from the SQL database"
    )
    needs_document_search: bool = Field(
        description=(
            "True if answering requires looking inside any ingested/uploaded document "
            "(policy PDFs or anything else the user uploaded), regardless of topic"
        )
    )


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    needs_customer_data: bool
    needs_document_search: bool
    sql_result: Optional[str]
    docs_result: Optional[str]
    final_answer: str


class SupportAssistant:
    """Wraps the compiled LangGraph multi-agent graph. Call `await setup()` once,
    then `await ask(messages)` per turn."""

    def __init__(self):
        self.client = MultiServerMCPClient(
            {"support_tools": {"transport": "streamable_http", "url": MCP_SERVER_URL}}
        )
        self.llm = get_chat_model()
        self.router_llm = self.llm.with_structured_output(
            RouteDecision, method=get_structured_output_method()
        )
        self.sql_agent = None
        self.docs_agent = None
        self.graph = None

    async def setup(self):
        tools = await self.client.get_tools()
        sql_tools = [t for t in tools if t.name in SQL_TOOL_NAMES]
        docs_tools = [t for t in tools if t.name in DOCS_TOOL_NAMES]
        self.sql_agent = create_react_agent(self.llm, sql_tools, prompt=SQL_AGENT_PROMPT)
        self.docs_agent = create_react_agent(self.llm, docs_tools, prompt=DOCS_AGENT_PROMPT)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("router", self._route)
        graph.add_node("sql_agent", self._run_sql_agent)
        graph.add_node("docs_agent", self._run_docs_agent)
        graph.add_node("synthesizer", self._synthesize)

        graph.add_edge(START, "router")
        graph.add_conditional_edges("router", self._dispatch, ["sql_agent", "docs_agent", "synthesizer"])
        graph.add_edge("sql_agent", "synthesizer")
        graph.add_edge("docs_agent", "synthesizer")
        graph.add_edge("synthesizer", END)
        return graph.compile()

    async def _route(self, state: AgentState) -> dict:
        history = state["messages"][-6:]
        decision: RouteDecision = await self.router_llm.ainvoke(
            [SystemMessage(content=ROUTER_PROMPT), *history]
        )
        return {
            "needs_customer_data": decision.needs_customer_data,
            "needs_document_search": decision.needs_document_search,
            "sql_result": None,
            "docs_result": None,
        }

    def _dispatch(self, state: AgentState) -> list[Literal["sql_agent", "docs_agent", "synthesizer"]]:
        dests: list[Literal["sql_agent", "docs_agent", "synthesizer"]] = []
        if state["needs_customer_data"]:
            dests.append("sql_agent")
        if state["needs_document_search"]:
            dests.append("docs_agent")
        return dests or ["synthesizer"]

    async def _run_sql_agent(self, state: AgentState) -> dict:
        result = await self.sql_agent.ainvoke({"messages": state["messages"]})
        return {"sql_result": result["messages"][-1].content}

    async def _run_docs_agent(self, state: AgentState) -> dict:
        result = await self.docs_agent.ainvoke({"messages": state["messages"]})
        return {"docs_result": result["messages"][-1].content}

    async def _synthesize(self, state: AgentState) -> dict:
        context_parts = []
        if state.get("sql_result"):
            context_parts.append(f"[Customer database findings]\n{state['sql_result']}")
        if state.get("docs_result"):
            context_parts.append(f"[Policy document findings]\n{state['docs_result']}")
        context = "\n\n".join(context_parts) if context_parts else "(no external data was needed for this message)"

        response = await self.llm.ainvoke(
            [SystemMessage(content=SYNTHESIS_PROMPT.format(context=context)), *state["messages"]]
        )
        return {"final_answer": response.content, "messages": [response]}

    async def ask(self, messages: list) -> str:
        """messages: list of LangChain message objects (conversation so far, including the
        latest HumanMessage)."""
        if self.graph is None:
            await self.setup()
        result = await self.graph.ainvoke({"messages": messages})
        return result["final_answer"]
