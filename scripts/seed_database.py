"""Generate a synthetic customer + support-ticket dataset and load it into SQLite.

Run:
    python scripts/seed_database.py
"""
import random
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faker import Faker

from agents.config import DATABASE_PATH

fake = Faker()
Faker.seed(42)
random.seed(42)

PLANS = ["Free", "Basic", "Pro", "Enterprise"]
STATUSES = ["Open", "In Progress", "Resolved", "Closed"]
PRIORITIES = ["Low", "Medium", "High", "Urgent"]
CATEGORIES = ["Billing", "Technical", "Refund", "Shipping", "Account", "General"]

SCHEMA = """
DROP TABLE IF EXISTS support_tickets;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id     TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    email           TEXT NOT NULL,
    phone           TEXT,
    plan            TEXT,
    account_status  TEXT,
    city            TEXT,
    country         TEXT,
    signup_date     TEXT
);

CREATE TABLE support_tickets (
    ticket_id           TEXT PRIMARY KEY,
    customer_id         TEXT NOT NULL,
    subject             TEXT NOT NULL,
    description         TEXT,
    category            TEXT,
    priority            TEXT,
    status              TEXT,
    created_at          TEXT,
    resolved_at         TEXT,
    resolution_notes    TEXT,
    satisfaction_score  INTEGER,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
"""

TICKET_TEMPLATES = {
    "Billing": [
        ("Double charge on last invoice", "Customer was billed twice for the {month} subscription cycle."),
        ("Question about upgrade proration", "Customer wants to understand proration when upgrading from {plan}."),
    ],
    "Technical": [
        ("App crashes on login", "Customer reports the mobile app crashes immediately after entering credentials."),
        ("Sync issue between devices", "Data is not syncing between customer's desktop and mobile app."),
    ],
    "Refund": [
        ("Refund request for annual plan", "Customer wants a refund after cancelling within the 30-day window."),
        ("Refund not received", "Customer approved for a refund 10 days ago but has not received the funds."),
    ],
    "Shipping": [
        ("Order delayed", "Customer's hardware order has not shipped after the promised window."),
        ("Wrong item received", "Customer received the wrong SKU in their shipment."),
    ],
    "Account": [
        ("Cannot reset password", "Customer is not receiving the password reset email."),
        ("Request to merge duplicate accounts", "Customer accidentally created two accounts with different emails."),
    ],
    "General": [
        ("Feature request: dark mode", "Customer is requesting a dark mode option in the dashboard."),
        ("General product feedback", "Customer shared feedback about onboarding experience."),
    ],
}


def random_date(start_days_ago=400, end_days_ago=1):
    delta = random.randint(end_days_ago, start_days_ago)
    return datetime.now() - timedelta(days=delta)


def make_customer(idx: int, name: str | None = None, email: str | None = None) -> dict:
    signup = random_date(900, 30)
    return {
        "customer_id": f"CUST-{1000 + idx}",
        "name": name or fake.name(),
        "email": email or fake.unique.email(),
        "phone": fake.phone_number(),
        "plan": random.choice(PLANS),
        "account_status": random.choices(["Active", "Suspended", "Churned"], weights=[80, 10, 10])[0],
        "city": fake.city(),
        "country": fake.country(),
        "signup_date": signup.strftime("%Y-%m-%d"),
    }


def make_ticket(idx: int, customer_id: str, signup_date: str) -> dict:
    category = random.choice(CATEGORIES)
    subject, desc_template = random.choice(TICKET_TEMPLATES[category])
    description = desc_template.format(month=fake.month_name(), plan=random.choice(PLANS))
    earliest = datetime.strptime(signup_date, "%Y-%m-%d")
    created = earliest + timedelta(days=random.randint(1, 300))
    if created > datetime.now():
        created = datetime.now() - timedelta(days=random.randint(0, 30))
    status = random.choices(STATUSES, weights=[10, 15, 40, 35])[0]
    resolved_at = None
    resolution_notes = None
    satisfaction = None
    if status in ("Resolved", "Closed"):
        resolved_at = (created + timedelta(days=random.randint(1, 10))).strftime("%Y-%m-%d %H:%M:%S")
        resolution_notes = random.choice(
            [
                "Issue reproduced and fixed in latest release.",
                "Refund processed successfully to original payment method.",
                "Walked customer through troubleshooting steps; resolved.",
                "Replacement item shipped at no additional cost.",
                "Escalated to engineering; hotfix deployed.",
            ]
        )
        satisfaction = random.randint(3, 5)
    return {
        "ticket_id": f"TCK-{2000 + idx}",
        "customer_id": customer_id,
        "subject": subject,
        "description": description,
        "category": category,
        "priority": random.choices(PRIORITIES, weights=[30, 35, 25, 10])[0],
        "status": status,
        "created_at": created.strftime("%Y-%m-%d %H:%M:%S"),
        "resolved_at": resolved_at,
        "resolution_notes": resolution_notes,
        "satisfaction_score": satisfaction,
    }


def main():
    Path(DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()
    cur.executescript(SCHEMA)

    customers = []
    # Featured customer used in the assignment's example query.
    customers.append(make_customer(0, name="Ema Thompson", email="ema.thompson@example.com"))
    for i in range(1, 40):
        customers.append(make_customer(i))

    cur.executemany(
        """INSERT INTO customers
           (customer_id, name, email, phone, plan, account_status, city, country, signup_date)
           VALUES (:customer_id, :name, :email, :phone, :plan, :account_status, :city, :country, :signup_date)""",
        customers,
    )

    tickets = []
    ticket_idx = 0
    signup_by_id = {c["customer_id"]: c["signup_date"] for c in customers}

    # Guarantee the featured customer has a rich, varied ticket history.
    for _ in range(5):
        tickets.append(make_ticket(ticket_idx, "CUST-1000", signup_by_id["CUST-1000"]))
        ticket_idx += 1

    for customer in customers[1:]:
        for _ in range(random.randint(1, 4)):
            tickets.append(make_ticket(ticket_idx, customer["customer_id"], signup_by_id[customer["customer_id"]]))
            ticket_idx += 1

    cur.executemany(
        """INSERT INTO support_tickets
           (ticket_id, customer_id, subject, description, category, priority, status,
            created_at, resolved_at, resolution_notes, satisfaction_score)
           VALUES (:ticket_id, :customer_id, :subject, :description, :category, :priority, :status,
                   :created_at, :resolved_at, :resolution_notes, :satisfaction_score)""",
        tickets,
    )

    conn.commit()
    n_customers = cur.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    n_tickets = cur.execute("SELECT COUNT(*) FROM support_tickets").fetchone()[0]
    conn.close()
    print(f"Seeded {DATABASE_PATH}: {n_customers} customers, {n_tickets} support tickets.")


if __name__ == "__main__":
    main()
