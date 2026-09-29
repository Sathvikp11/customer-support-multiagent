"""Streamlit UI for the customer-support multi-agent system.

Run:
    streamlit run app/streamlit_app.py
(Requires the MCP server to already be running: python -m mcp_server.server)
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agents.config import LLM_PROVIDER, MCP_SERVER_URL, OLLAMA_MODEL, OPENAI_MODEL, POLICY_DOCS_DIR
from agents.graph import SupportAssistant
from mcp_server.knowledge_base import ingest_pdf

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
            st.text(f"• {doc.name}")
    else:
        st.caption("No documents ingested yet.")

    uploaded_files = st.file_uploader(
        "Upload a company policy PDF", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files and st.button("Ingest uploaded PDF(s)"):
        docs_dir.mkdir(parents=True, exist_ok=True)
        with st.spinner("Chunking, embedding, and indexing document(s)..."):
            for uploaded in uploaded_files:
                dest = docs_dir / uploaded.name
                dest.write_bytes(uploaded.getvalue())
                n_chunks = ingest_pdf(dest)
                st.success(f"Ingested {uploaded.name}: {n_chunks} chunks")
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
