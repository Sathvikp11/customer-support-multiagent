"""Ingest all policy PDFs in data/policies/ into the Chroma vector store.

Run:
    python scripts/ingest_documents.py [--reset]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from agents.config import MCP_SERVER_URL
from mcp_server.knowledge_base import ingest_directory, reset_collection


def _mcp_server_running() -> bool:
    try:
        requests.get(MCP_SERVER_URL, timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Wipe the collection before ingesting")
    args = parser.parse_args()

    if _mcp_server_running():
        print(
            "The MCP server is currently running. This script writes to the vector store "
            "directly, and running it at the same time as the server can corrupt the "
            "server's in-memory view of the collection (two processes writing to the same "
            "embedded Chroma directory is not safe). Either stop the MCP server first, or "
            "use the 'Upload a policy PDF' feature in the Streamlit app instead, which "
            "delegates ingestion through the running server safely."
        )
        sys.exit(1)

    if args.reset:
        print("Resetting existing collection...")
        reset_collection()

    results = ingest_directory()
    if not results:
        print("No PDFs found in data/policies/. Run scripts/generate_policy_pdfs.py first.")
        return
    total = 0
    for name, count in results.items():
        print(f"  {name}: {count} chunks")
        total += count
    print(f"Ingested {total} chunks from {len(results)} document(s) into the vector store.")


if __name__ == "__main__":
    main()
