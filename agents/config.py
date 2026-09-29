"""Shared configuration loaded from environment / .env for all components
(MCP server, ingestion scripts, agents, Streamlit app)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

DATABASE_PATH = str(ROOT_DIR / os.getenv("DATABASE_PATH", "data/support.db"))
CHROMA_DIR = str(ROOT_DIR / os.getenv("CHROMA_DIR", "data/chroma_db"))
POLICY_DOCS_DIR = str(ROOT_DIR / os.getenv("POLICY_DOCS_DIR", "data/policies"))

MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8765"))
MCP_SERVER_URL = f"http://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}/mcp"


def get_chat_model(temperature: float = 0.0):
    """Return a configured LangChain chat model based on LLM_PROVIDER."""
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=ANTHROPIC_MODEL, temperature=temperature, api_key=ANTHROPIC_API_KEY)

    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=OLLAMA_MODEL, temperature=temperature, base_url=OLLAMA_BASE_URL)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=OPENAI_MODEL, temperature=temperature, api_key=OPENAI_API_KEY)


def get_structured_output_method() -> str:
    """Ollama's tool-calling mode doesn't reliably route JSON through tool_calls for
    small local models, so use its native json_schema structured-output mode there;
    OpenAI/Anthropic use the default function-calling method."""
    return "json_schema" if LLM_PROVIDER == "ollama" else "function_calling"


def get_embeddings():
    """Local, key-free embedding model shared by ingestion and retrieval."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
