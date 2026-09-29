"""Streamlit UI for the customer-support multi-agent system.

Run:
    streamlit run app/streamlit_app.py
(Requires the MCP server to already be running: python -m mcp_server.server)
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from agents.config import LLM_PROVIDER, MCP_SERVER_URL, OLLAMA_MODEL, OPENAI_MODEL, POLICY_DOCS_DIR
from agents.graph import SupportAssistant

st.set_page_config(page_title="Customer Support Copilot", page_icon="🎧", layout="wide")


def mcp_server_reachable() -> bool:
    try:
        # The MCP streamable-http endpoint responds (with a redirect/406) even to a bare GET;
        # any HTTP response at all means the server process is up.
        requests.get(MCP_SERVER_URL, timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False


async def run_turn(messages: list) -> str:
    assistant = SupportAssistant()
    await assistant.setup()
    return await assistant.ask(messages)


def ask_assistant(messages: list) -> str:
    return asyncio.run(run_turn(messages))


async def _reingest_via_mcp() -> dict:
    # Ingestion is delegated to the MCP server's own tool rather than touching the Chroma
    # store from this process directly: Chroma's embedded mode isn't safe for two separate
    # OS processes (this Streamlit app and the long-running MCP server) to write to the
    # same persisted collection concurrently, so the MCP server must be the sole writer.
    client = MultiServerMCPClient(
        {"support_tools": {"transport": "streamable_http", "url": MCP_SERVER_URL}}
    )
    tools = await client.get_tools()
    tool = next(t for t in tools if t.name == "reingest_policy_documents")
    result = await tool.ainvoke({})
    return json.loads(result)


def reingest_documents() -> dict:
    return asyncio.run(_reingest_via_mcp())


async def _remove_via_mcp(filename: str) -> dict:
    client = MultiServerMCPClient(
        {"support_tools": {"transport": "streamable_http", "url": MCP_SERVER_URL}}
    )
    tools = await client.get_tools()
    tool = next(t for t in tools if t.name == "remove_policy_document")
    result = await tool.ainvoke({"filename": filename})
    return json.loads(result)


def remove_document(filename: str) -> dict:
    return asyncio.run(_remove_via_mcp(filename))


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list[HumanMessage | AIMessage]

with st.sidebar:
    st.header("🎧 Support Copilot")
    st.caption("Generative AI multi-agent assistant over structured (SQL) and unstructured (PDF) data.")

    if mcp_server_reachable():
        st.success("MCP server: connected")
    else:
        st.error("MCP server: unreachable")
        st.code("python -m mcp_server.server", language="bash")
        st.caption("Start it in a separate terminal, then reload this page.")

    model_label = {"ollama": OLLAMA_MODEL, "openai": OPENAI_MODEL, "anthropic": "Claude"}.get(
        LLM_PROVIDER, LLM_PROVIDER
    )
    st.caption(f"LLM provider: **{LLM_PROVIDER}** ({model_label})")

    st.divider()
    st.subheader("📄 Policy knowledge base")
    docs_dir = Path(POLICY_DOCS_DIR)
    existing_docs = sorted(docs_dir.glob("*.pdf")) if docs_dir.exists() else []
    if existing_docs:
        for doc in existing_docs:
            doc_col, remove_col = st.columns([5, 1])
            doc_col.text(doc.name)
            if remove_col.button("🗑", key=f"remove_{doc.name}", help=f"Remove {doc.name}"):
                with st.spinner(f"Removing {doc.name}..."):
                    try:
                        remove_document(doc.name)
                    except Exception as exc:
                        st.error(f"Failed to remove: {exc}")
                st.rerun()
    else:
        st.caption("No documents ingested yet.")

    uploaded_files = st.file_uploader(
        "Upload a company policy PDF", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files and st.button("Ingest uploaded PDF(s)"):
        docs_dir.mkdir(parents=True, exist_ok=True)
        for uploaded in uploaded_files:
            (docs_dir / uploaded.name).write_bytes(uploaded.getvalue())
        with st.spinner("Asking the MCP server to chunk, embed, and index the document(s)..."):
            try:
                results = reingest_documents()
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")
                results = {}
        for name, n_chunks in results.items():
            st.success(f"Ingested {name}: {n_chunks} chunks")
        st.rerun()

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()

st.title("Customer Support Copilot")
st.caption(
    "Ask about company policies (refund, privacy, shipping, SLA) or about a specific "
    "customer's profile and support ticket history — the multi-agent system will figure "
    "out which data source(s) to use."
)

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

example_cols = st.columns(2)
example_questions = [
    "What is the current refund policy?",
    "Give me a quick overview of customer Ema's profile and past support ticket details.",
]
clicked_example = None
for col, question in zip(example_cols, example_questions):
    if col.button(question, use_container_width=True):
        clicked_example = question

user_input = st.chat_input("Ask about a policy or a customer...")
question = clicked_example or user_input

if question:
    st.session_state.chat_history.append(HumanMessage(content=question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not mcp_server_reachable():
            answer = (
                "The MCP tool server isn't running, so I can't reach the customer database "
                "or the policy knowledge base. Start it with `python -m mcp_server.server` "
                "and try again."
            )
            st.error(answer)
        else:
            with st.spinner("Thinking..."):
                try:
                    answer = ask_assistant(st.session_state.chat_history)
                except Exception as exc:  # surfaced to the user rather than a blank crash
                    answer = f"Something went wrong answering that: {exc}"
            st.markdown(answer)

    st.session_state.chat_history.append(AIMessage(content=answer))
