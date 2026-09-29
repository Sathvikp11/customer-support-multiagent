"""Shared vector-store access for the unstructured policy-document knowledge base.

Used by:
- scripts/ingest_documents.py (CLI ingestion of data/policies/*.pdf)
- app/streamlit_app.py (ingesting PDFs John uploads at runtime)
- mcp_server/server.py (the search_policy_documents MCP tool)
"""
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agents.config import CHROMA_DIR, POLICY_DOCS_DIR, get_embeddings

COLLECTION_NAME = "policy_documents"

_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

_vector_store = None


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_DIR,
        )
    return _vector_store


def ingest_pdf(path: Path) -> int:
    """Load, chunk, embed, and persist a single PDF. Returns number of chunks added.

    Idempotent: re-ingesting a file with the same name first removes its previously
    ingested chunks, so re-uploading a document never creates duplicates."""
    source_name = Path(path).name
    store = get_vector_store()
    existing = store.get(where={"source": source_name})
    if existing["ids"]:
        store.delete(ids=existing["ids"])

    loader = PyPDFLoader(str(path))
    pages = loader.load()
    for page in pages:
        page.metadata["source"] = source_name
    chunks = _splitter.split_documents(pages)
    if not chunks:
        return 0
    store.add_documents(chunks)
    return len(chunks)


def ingest_directory(directory: str = POLICY_DOCS_DIR) -> dict:
    """Ingest every PDF found in a directory. Returns a {filename: chunk_count} summary."""
    results = {}
    for pdf_path in sorted(Path(directory).glob("*.pdf")):
        results[pdf_path.name] = ingest_pdf(pdf_path)
    return results


def reset_collection():
    """Drop and recreate the collection (used when re-ingesting from scratch)."""
    global _vector_store
    store = get_vector_store()
    store.delete_collection()
    _vector_store = None


def search(query: str, k: int = 4) -> list[dict]:
    store = get_vector_store()
    results = store.similarity_search_with_score(query, k=k)
    return [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page"),
            "content": doc.page_content,
            "score": float(score),
        }
        for doc, score in results
    ]
