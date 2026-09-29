"""Generate a set of dummy company policy PDF documents used as the
unstructured knowledge base for the RAG agent.

Run:
    python scripts/generate_policy_pdfs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

from agents.config import POLICY_DOCS_DIR

STYLES = getSampleStyleSheet()
TITLE_STYLE = ParagraphStyle("TitleStyle", parent=STYLES["Title"], fontSize=20, spaceAfter=18)
HEADING_STYLE = ParagraphStyle("HeadingStyle", parent=STYLES["Heading2"], spaceBefore=14, spaceAfter=8)
BODY_STYLE = ParagraphStyle("BodyStyle", parent=STYLES["BodyText"], fontSize=10.5, leading=15, spaceAfter=8)

DOCUMENTS = {
    "refund_policy.pdf": {
        "title": "Acme Cloud Refund Policy",
        "sections": [
            ("Overview",
             "Acme Cloud ('Acme', 'we', 'us') wants every customer to be satisfied with our products. "
             "This Refund Policy explains when and how customers can request a refund for subscription "
             "purchases, one-time purchases, and hardware add-ons."),
            ("30-Day Money-Back Guarantee",
             "New customers on any paid plan (Basic, Pro, or Enterprise) may request a full refund within "
             "30 calendar days of their initial purchase, provided the account has not exceeded 20 GB of "
             "usage or 10,000 API calls during the trial window. Refund requests after 30 days are handled "
             "on a case-by-case basis by the support team."),
            ("Annual Plan Cancellations",
             "Customers who cancel an annual subscription within the first 30 days receive a full refund. "
             "Cancellations between day 31 and day 90 are eligible for a prorated refund covering the "
             "unused portion of the year, minus a 10% early-cancellation processing fee. No refunds are "
             "issued for annual plans cancelled after 90 days; service will remain active until the end of "
             "the paid term."),
            ("Refunds for Billing Errors",
             "If a customer is charged in error (for example, a duplicate charge or a charge after proper "
             "cancellation), Acme will issue a full refund of the erroneous charge within 5-7 business days "
             "of the issue being confirmed by our billing team, with no processing fee."),
            ("Non-Refundable Items",
             "The following are non-refundable: custom onboarding or professional services fees, add-on "
             "credits that have already been consumed, and Enterprise contract setup fees, unless otherwise "
             "stated in a signed order form."),
            ("How to Request a Refund",
             "Customers can request a refund by opening a support ticket under the 'Billing' category or by "
             "emailing billing@acmecloud.example. Please include the account email, invoice number, and "
             "reason for the request. Approved refunds are returned to the original payment method within "
             "5-10 business days."),
        ],
    },
    "privacy_policy.pdf": {
        "title": "Acme Cloud Privacy Policy",
        "sections": [
            ("Information We Collect",
             "We collect account information (name, email, company), usage data (feature interactions, "
             "log-in timestamps), and support interaction data (tickets, chat transcripts) to operate and "
             "improve our services."),
            ("How We Use Information",
             "Customer data is used to provide the service, respond to support requests, personalize the "
             "product experience, detect fraud and abuse, and comply with legal obligations. We do not sell "
             "customer personal data to third parties."),
            ("Data Retention",
             "Account and billing records are retained for 7 years to satisfy tax and accounting "
             "requirements. Support ticket data is retained for 3 years after ticket closure. Customers may "
             "request earlier deletion subject to legal retention requirements."),
            ("Data Sharing",
             "We share data with sub-processors strictly necessary to deliver the service (e.g., cloud "
             "hosting, payment processing, email delivery), each bound by a data processing agreement. We "
             "do not share data with advertisers."),
            ("Customer Rights",
             "Depending on your jurisdiction, you may have the right to access, correct, export, or delete "
             "your personal data. Requests can be submitted via the privacy@acmecloud.example mailbox and "
             "are processed within 30 days."),
            ("Security",
             "Data is encrypted in transit (TLS 1.2+) and at rest (AES-256). Access to production data is "
             "restricted to authorized personnel and logged for audit purposes."),
        ],
    },
    "shipping_policy.pdf": {
        "title": "Acme Cloud Hardware Shipping Policy",
        "sections": [
            ("Order Processing",
             "Hardware orders (edge devices, connector kits) are processed within 1-2 business days of "
             "payment confirmation. Orders placed after 2 PM local warehouse time are processed the next "
             "business day."),
            ("Shipping Timelines",
             "Standard shipping within the continental US takes 3-5 business days. Expedited shipping takes "
             "1-2 business days. International shipping typically takes 7-14 business days depending on "
             "customs processing in the destination country."),
            ("Shipping Costs",
             "Standard shipping is free for Pro and Enterprise plans and $9.99 for Free and Basic plans. "
             "Expedited and international shipping costs are calculated at checkout based on destination "
             "and package weight."),
            ("Order Tracking",
             "A tracking number is emailed to the customer once the order ships. Tracking status can also "
             "be viewed from the 'Orders' tab in the customer dashboard."),
            ("Damaged or Incorrect Items",
             "If a shipment arrives damaged or contains the wrong item, customers should open a support "
             "ticket under the 'Shipping' category within 14 days of delivery, including photos of the "
             "item and packaging. Acme will ship a free replacement and, where applicable, arrange return "
             "of the incorrect item at no cost to the customer."),
            ("Delayed Shipments",
             "If a shipment has not arrived within the estimated window, customers can contact support for "
             "a status update. Shipments delayed more than 5 business days beyond the estimated delivery "
             "date qualify for a shipping fee refund upon request."),
        ],
    },
    "support_sla_policy.pdf": {
        "title": "Acme Cloud Customer Support SLA",
        "sections": [
            ("Support Channels",
             "Customers can reach support via the in-app chat assistant, email, or the community forum. "
             "Enterprise customers additionally have access to a dedicated Slack channel and phone support."),
            ("Priority Levels",
             "Tickets are classified as Urgent, High, Medium, or Low based on business impact. Urgent "
             "issues involve full service outages; High issues involve major feature degradation; Medium "
             "issues involve minor bugs or how-to questions; Low issues involve feature requests or general "
             "feedback."),
            ("First Response Targets",
             "Urgent: 1 hour (Enterprise), 4 hours (Pro), 8 hours (Basic/Free). High: 4 hours (Enterprise), "
             "8 hours (Pro), 24 hours (Basic/Free). Medium: 1 business day for all plans. Low: 2 business "
             "days for all plans."),
            ("Resolution Targets",
             "Urgent issues are targeted for resolution or a workaround within 8 business hours. High "
             "priority issues within 2 business days. Medium priority issues within 5 business days. Low "
             "priority items are addressed on a best-effort basis and may be added to the product roadmap."),
            ("Escalation Process",
             "If a customer feels their issue is not progressing according to these targets, they can "
             "request escalation by replying 'ESCALATE' on the ticket or contacting their account manager "
             "directly. Escalated tickets are reviewed by a support lead within 2 business hours."),
            ("Customer Satisfaction Follow-up",
             "After a ticket is resolved, customers receive a short satisfaction survey (1-5 scale). Scores "
             "of 3 or below trigger an automatic follow-up from a support lead to ensure the issue was "
             "fully addressed."),
        ],
    },
}


def build_pdf(path: Path, title: str, sections: list[tuple[str, str]]):
    doc = SimpleDocTemplate(str(path), pagesize=LETTER, topMargin=0.9 * inch, bottomMargin=0.9 * inch)
    story = [Paragraph(title, TITLE_STYLE), Spacer(1, 6)]
    for heading, body in sections:
        story.append(Paragraph(heading, HEADING_STYLE))
        story.append(Paragraph(body, BODY_STYLE))
    doc.build(story)


def main():
    out_dir = Path(POLICY_DOCS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in DOCUMENTS.items():
        target = out_dir / filename
        build_pdf(target, content["title"], content["sections"])
        print(f"Generated {target}")


if __name__ == "__main__":
    main()
