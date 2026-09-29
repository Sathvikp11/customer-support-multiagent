"""MCP server exposing the structured (SQL) and unstructured (vector search)
data sources as tools that any MCP-compatible agent can call.

Run standalone:
    python -m mcp_server.server
Serves streamable-HTTP at http://<host>:<port>/mcp (see agents/config.py).
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP

from agents.config import DATABASE_PATH, MCP_SERVER_HOST, MCP_SERVER_PORT
from mcp_server.knowledge_base import ingest_directory, search

mcp = FastMCP("customer-support-tools", host=MCP_SERVER_HOST, port=MCP_SERVER_PORT)

SCHEMA_DESCRIPTION = """\
Table customers(
    customer_id TEXT PRIMARY KEY,   -- e.g. 'CUST-1000'
    name TEXT,
    email TEXT,
    phone TEXT,
    plan TEXT,                      -- 'Free' | 'Basic' | 'Pro' | 'Enterprise'
    account_status TEXT,            -- 'Active' | 'Suspended' | 'Churned'
    city TEXT,
    country TEXT,
    signup_date TEXT                -- 'YYYY-MM-DD'
)

Table support_tickets(
    ticket_id TEXT PRIMARY KEY,     -- e.g. 'TCK-2000'
    customer_id TEXT,               -- foreign key -> customers.customer_id
    subject TEXT,
    description TEXT,
    category TEXT,                  -- 'Billing' | 'Technical' | 'Refund' | 'Shipping' | 'Account' | 'General'
    priority TEXT,                  -- 'Low' | 'Medium' | 'High' | 'Urgent'
    status TEXT,                    -- 'Open' | 'In Progress' | 'Resolved' | 'Closed'
    created_at TEXT,                -- 'YYYY-MM-DD HH:MM:SS'
    resolved_at TEXT,                -- nullable
    resolution_notes TEXT,           -- nullable
    satisfaction_score INTEGER       -- 1-5, nullable
)
"""

BLOCKED_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "attach", "detach",
    "pragma", "create", "replace", "vacuum", "reindex", "--", "/*",
)


def _rows_to_dicts(cursor: sqlite3.Cursor, rows: list[sqlite3.Row]) -> list[dict]:
    cols = [c[0] for c in cursor.description]
    return [dict(zip(cols, row)) for row in rows]


@mcp.tool()
def get_database_schema() -> str:
    """Return the SQL schema of the structured customer database (tables and columns)
    so you can write correct SELECT queries against it."""
    return SCHEMA_DESCRIPTION


@mcp.tool()
def run_sql_query(sql: str) -> str:
    """Execute a READ-ONLY SQL SELECT query against the customer support SQLite database
    and return the results as a list of row objects (JSON). Only SELECT statements are
    permitted. Call get_database_schema first if you are unsure of table/column names."""
    import json

    normalized = sql.strip().lower()
    if not normalized.startswith("select"):
        return json.dumps({"error": "Only SELECT statements are allowed."})
    if any(word in normalized for word in BLOCKED_KEYWORDS):
        return json.dumps({"error": "Query contains a disallowed keyword."})
    if ";" in sql.strip().rstrip(";"):
        return json.dumps({"error": "Only a single statement is allowed."})

    try:
        uri = f"file:{DATABASE_PATH}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchmany(200)
        result = _rows_to_dicts(cur, rows)
        conn.close()
        return json.dumps(result, default=str)
    except sqlite3.Error as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def get_customer_profile(name_or_id: str) -> str:
    """Look up a customer by name (partial match, case-insensitive) or exact customer_id,
    and return their profile plus their full support ticket history as JSON. This is the
    fastest way to answer 'give me an overview of customer X'."""
    import json

    conn = sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM customers WHERE customer_id = ? OR LOWER(name) LIKE ?",
        (name_or_id, f"%{name_or_id.lower()}%"),
    )
    customers = [dict(r) for r in cur.fetchall()]
    if not customers:
        conn.close()
        return json.dumps({"error": f"No customer found matching '{name_or_id}'."})

    for customer in customers:
        cur.execute(
            "SELECT ticket_id, subject, category, priority, status, created_at, "
            "resolved_at, resolution_notes, satisfaction_score FROM support_tickets "
            "WHERE customer_id = ? ORDER BY created_at DESC",
            (customer["customer_id"],),
        )
        customer["tickets"] = [dict(r) for r in cur.fetchall()]
    conn.close()
    return json.dumps(customers, default=str)


@mcp.tool()
def list_customers(query: str = "", limit: int = 20) -> str:
    """List customers, optionally filtered by a name/email substring. Useful for
    disambiguating when a user query is ambiguous (e.g. multiple customers named 'Sam')."""
    import json

    conn = sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    like = f"%{query.lower()}%"
    cur.execute(
        "SELECT customer_id, name, email, plan, account_status FROM customers "
        "WHERE LOWER(name) LIKE ? OR LOWER(email) LIKE ? LIMIT ?",
        (like, like, limit),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return json.dumps(rows, default=str)


@mcp.tool()
def search_policy_documents(query: str, k: int = 4) -> str:
    """Semantic search over the ingested company policy PDFs (refund, privacy, shipping,
    support SLA, and any documents uploaded via the UI). Returns the most relevant text
    chunks with their source document name so you can cite them in your answer."""
    import json

    results = search(query, k=k)
    if not results:
        return json.dumps({"error": "No documents found. Has anything been ingested yet?"})
    return json.dumps(results, default=str)


@mcp.tool()
def reingest_policy_documents() -> str:
    """Re-scan the policy documents folder and ingest any PDFs into the vector store.
    Call this after a new document has been uploaded to the knowledge base."""
    import json

    results = ingest_directory()
    return json.dumps(results, default=str)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
