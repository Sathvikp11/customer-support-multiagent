"""Ingest all policy PDFs in data/policies/ into the Chroma vector store.

Run:
    python scripts/ingest_documents.py [--reset]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_server.knowledge_base import ingest_directory, reset_collection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Wipe the collection before ingesting")
    args = parser.parse_args()

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
