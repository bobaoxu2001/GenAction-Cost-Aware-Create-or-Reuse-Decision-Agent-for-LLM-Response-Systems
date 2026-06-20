"""Generate the sample customer-support dataset used by GenAction.

This script is the *provenance* of the two CSV files under ``data/``. The CSVs
are committed to the repository, so a reviewer never has to run this script to
use the project. It is kept here so the dataset is fully reproducible and the
design intent is documented in code.

Design intent
-------------
The stream is deliberately *heterogeneous in coverage* so that the
create-or-reuse trade-off is non-trivial:

* **Well-covered categories** (billing, refund, account_login, troubleshooting,
  pricing, onboarding) already have good answers in the FAQ library. Incoming
  queries are paraphrases of those FAQs, so *reuse* is cheap and *creating* a
  new action mostly wastes the creation budget.

* **Poorly-covered categories** (subscription_cancel, sales_followup,
  crm_update, meeting_summary) have *no* FAQ entry. The first query in such a
  cluster is a poor fit for everything in the library, so *creating* a reusable
  action is worth its cost -- it is then amortised over later, similar queries.

A good adaptive policy should therefore create selectively (for the
poorly-covered clusters) and reuse otherwise, beating both naive baselines and
a policy that creates at a fixed random rate.

Run with::

    python scripts/make_dataset.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEED = 7

# --------------------------------------------------------------------------- #
# FAQ library: initial reusable actions (well-covered categories only).
# columns: action_id, category, canonical_query, response_text
# --------------------------------------------------------------------------- #
FAQ_LIBRARY = [
    (
        "billing",
        "Why was I charged twice for my subscription this month?",
        "It looks like a duplicate charge may have occurred. We'll review your "
        "billing history and refund any duplicate payment within 5 to 7 "
        "business days.",
    ),
    (
        "billing",
        "How do I update the credit card used for billing?",
        "You can update your payment method under Settings > Billing > Payment "
        "Methods, where you can add or replace your card.",
    ),
    (
        "refund",
        "How do I request a refund for my recent order?",
        "We're sorry to hear that. Open Orders, select the order, and choose "
        "Request Refund. We'll process it within 5 to 7 business days.",
    ),
    (
        "refund",
        "What is your refund policy for digital downloads?",
        "Digital products can be refunded within 14 days of purchase if they "
        "have not been substantially downloaded or used.",
    ),
    (
        "account_login",
        "I cannot log into my account, what should I do?",
        "Please try resetting your password with the Forgot Password link. If "
        "you still cannot log in, clear your browser cache or contact us to "
        "unlock the account.",
    ),
    (
        "account_login",
        "How do I reset my forgotten password?",
        "Click Forgot Password on the login page, enter your email, and follow "
        "the secure reset link we send you.",
    ),
    (
        "troubleshooting",
        "The mobile app keeps crashing on startup, how do I fix it?",
        "Please update to the latest version, restart your device, and "
        "reinstall the app if crashes continue. Sharing the error log helps us "
        "diagnose faster.",
    ),
    (
        "troubleshooting",
        "My dashboard is not loading any data.",
        "Try refreshing the page and clearing the cache. If the dashboard is "
        "still blank, check your network connection and any active filters that "
        "might hide your data.",
    ),
    (
        "pricing",
        "How much does the Pro plan cost each month?",
        "The Pro plan is $29 per user per month billed monthly, or $290 per "
        "user per year when billed annually.",
    ),
    (
        "pricing",
        "What is the difference between the Basic and Pro plans?",
        "Basic covers core features for small teams, while Pro adds advanced "
        "analytics, integrations, and priority support.",
    ),
    (
        "onboarding",
        "How do I get started after I sign up?",
        "Welcome aboard! Start by completing your profile, inviting teammates, "
        "and following the in-app setup checklist to configure your workspace.",
    ),
    (
        "onboarding",
        "How do I invite my teammates to the workspace?",
        "Open Settings > Members > Invite, enter their email addresses, and "
        "assign roles. They will receive an invite link by email.",
    ),
]

# --------------------------------------------------------------------------- #
# Stream: incoming queries. Grouped by (category, gold_response) cluster so the
# amortisation structure is explicit. Well-covered clusters reuse FAQ answers;
# poorly-covered clusters require creation.
# --------------------------------------------------------------------------- #

# gold responses reused by several queries inside a cluster
G = {
    # well-covered
    "bill_dup": "It looks like a duplicate charge may have occurred. We'll "
    "review your billing history and refund any duplicate payment within 5 to 7 "
    "business days.",
    "bill_card": "You can update your payment method under Settings > Billing > "
    "Payment Methods, where you can add or replace your card.",
    "refund_order": "We're sorry to hear that. Open Orders, select the order, "
    "and choose Request Refund. Refunds are processed within 5 to 7 business "
    "days.",
    "refund_digital": "Digital products can be refunded within 14 days of "
    "purchase if they have not been substantially downloaded or used.",
    "login": "Please try resetting your password with the Forgot Password link. "
    "If you still cannot log in, clear your browser cache or contact us to "
    "unlock the account.",
    "reset": "Click Forgot Password on the login page, enter your email, and "
    "follow the secure reset link we send you.",
    "crash": "Please update to the latest version, restart your device, and "
    "reinstall the app if crashes continue. Sharing the error log helps us "
    "diagnose faster.",
    "dashboard": "Try refreshing the page and clearing the cache. If the "
    "dashboard is still blank, check your network connection and any active "
    "filters that might hide your data.",
    "price_pro": "The Pro plan is $29 per user per month billed monthly, or "
    "$290 per user per year when billed annually.",
    "plan_diff": "Basic covers core features for small teams, while Pro adds "
    "advanced analytics, integrations, and priority support.",
    "start": "Welcome aboard! Start by completing your profile, inviting "
    "teammates, and following the in-app setup checklist to configure your "
    "workspace.",
    "invite": "Open Settings > Members > Invite, enter their email addresses, "
    "and assign roles. They will receive an invite link by email.",
    # poorly-covered (no FAQ -> must be created)
    "cancel": "You can cancel anytime under Settings > Subscription > Cancel "
    "Plan. Your access remains active until the end of the current billing "
    "period.",
    "pause": "Yes, you can pause your subscription for up to 3 months under "
    "Settings > Subscription > Pause. Billing stops while paused and resumes "
    "automatically.",
    "sales_followup": "Absolutely. I'll have an account executive reach out "
    "within one business day to discuss enterprise options and next steps.",
    "sales_quote": "Of course. I'll prepare a formal quote for your seat count "
    "and email it to you today, including volume discounts.",
    "crm_contact": "Sure, I've updated the contact record with the new details. "
    "Please allow a few minutes for it to sync across the system.",
    "crm_note": "Done. I've logged a note on your account recording today's "
    "conversation with a timestamp.",
    "meet_summary": "Here's a summary: we reviewed the project timeline, agreed "
    "on the next milestones, and assigned owners for each task. Let me know if "
    "I missed anything.",
    "meet_actions": "Action items: 1) Send the revised proposal by Friday, "
    "2) Schedule a follow-up call next week, 3) Share the onboarding docs with "
    "the team.",
}

# (category, gold_key, [query paraphrases])
#
# Each cluster is anchored on shared content words (e.g. "cancel subscription",
# "summarize meeting") so that paraphrases inside a cluster are close in the
# TF-IDF space and the two intents inside a category stay separable. Well-covered
# clusters share their anchor with the matching FAQ canonical query; poorly-
# covered clusters share no anchor with any FAQ, so they read as genuinely novel.
STREAM_CLUSTERS = [
    # ----- well-covered: reuse should win -----
    ("billing", "bill_dup", [
        "I was charged twice for my subscription this month, why?",
        "Why was I charged twice on my subscription this month?",
        "I see I was charged twice this month, can you help?",
    ]),
    ("billing", "bill_card", [
        "How do I update the card used for billing?",
        "I need to update my billing card to a new one.",
    ]),
    ("refund", "refund_order", [
        "How do I request a refund for my order?",
        "I want to request a refund for my recent order.",
        "Can I request a refund for my order from last week?",
    ]),
    ("refund", "refund_digital", [
        "What is the refund policy for digital downloads?",
        "Are digital downloads eligible for a refund?",
    ]),
    ("account_login", "login", [
        "I cannot log into my account, please help.",
        "Why can't I log into my account?",
        "I am unable to log into my account today.",
    ]),
    ("account_login", "reset", [
        "How do I reset my password?",
        "I need to reset my forgotten password.",
    ]),
    ("troubleshooting", "crash", [
        "The app keeps crashing on startup.",
        "Why does the app keep crashing when I open it?",
        "My app keeps crashing on startup, how do I fix it?",
    ]),
    ("troubleshooting", "dashboard", [
        "The dashboard is not loading any data.",
        "Why is my dashboard not loading data?",
    ]),
    ("pricing", "price_pro", [
        "How much does the Pro plan cost per month?",
        "What is the monthly cost of the Pro plan?",
        "How much per month is the Pro plan?",
    ]),
    ("pricing", "plan_diff", [
        "What is the difference between the Basic and Pro plans?",
        "How do the Basic and Pro plans differ?",
    ]),
    ("onboarding", "start", [
        "How do I get started after signing up?",
        "What do I do to get started after I sign up?",
        "How do I get started once I sign up?",
    ]),
    ("onboarding", "invite", [
        "How do I invite teammates to the workspace?",
        "How can I invite my teammates to the workspace?",
    ]),
    # ----- poorly-covered: create-then-reuse should win -----
    ("subscription_cancel", "cancel", [
        "How do I cancel my subscription?",
        "I want to cancel my subscription.",
        "How can I cancel my subscription?",
        "Please help me cancel my subscription.",
        "Where do I cancel my subscription?",
    ]),
    ("subscription_cancel", "pause", [
        "How do I pause my subscription?",
        "Can I pause my subscription for a while?",
        "I want to pause my subscription temporarily.",
        "How can I pause my subscription instead?",
    ]),
    ("sales_followup", "sales_followup", [
        "Can someone follow up with me about the enterprise plan?",
        "Please have a sales rep follow up about the enterprise plan.",
        "I would like a follow up about the enterprise plan.",
        "Can you follow up with me on enterprise pricing?",
        "Who can follow up with me about an enterprise plan?",
    ]),
    ("sales_followup", "sales_quote", [
        "Can you send me a quote for enterprise seats?",
        "I need a quote for enterprise seats.",
        "Please send a quote for 50 enterprise seats.",
        "Can I get a quote for enterprise seats?",
    ]),
    ("crm_update", "crm_contact", [
        "Can you update the contact record for my company?",
        "Please update the contact record on our account.",
        "I need to update our contact record.",
        "Update the contact record with new details, please.",
        "Can you update our company contact record?",
    ]),
    ("crm_update", "crm_note", [
        "Please log a note on our account about today's call.",
        "Can you log a note on our account?",
        "Log a note on our account that we spoke today.",
        "Please add and log a note on our account.",
    ]),
    ("meeting_summary", "meet_summary", [
        "Can you summarize our meeting today?",
        "Please summarize the meeting for me.",
        "Can you summarize the meeting notes?",
        "Summarize what happened in the meeting.",
        "Can you give me a summary of the meeting?",
    ]),
    ("meeting_summary", "meet_actions", [
        "What are the action items from the meeting?",
        "Please list the action items from our meeting.",
        "Can you share the action items from the meeting?",
        "What action items came out of the meeting?",
    ]),
]


def build_stream_rows() -> list[tuple[str, str, str]]:
    """Flatten clusters into (category, query_text, gold_response) and shuffle."""
    rows: list[tuple[str, str, str]] = []
    for category, gold_key, queries in STREAM_CLUSTERS:
        gold = G[gold_key]
        for q in queries:
            rows.append((category, q, gold))

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(rows))
    return [rows[i] for i in order]


def write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    faq_rows = [
        (f"FAQ-{i + 1:03d}", cat, q, resp)
        for i, (cat, q, resp) in enumerate(FAQ_LIBRARY)
    ]
    write_csv(
        DATA_DIR / "sample_faq_library.csv",
        ["action_id", "category", "canonical_query", "response_text"],
        faq_rows,
    )

    stream = build_stream_rows()
    stream_rows = [
        (f"Q{i + 1:03d}", cat, q, gold)
        for i, (cat, q, gold) in enumerate(stream)
    ]
    write_csv(
        DATA_DIR / "sample_stream.csv",
        ["query_id", "category", "query_text", "gold_response"],
        stream_rows,
    )

    print(f"Wrote {len(faq_rows)} FAQ actions -> data/sample_faq_library.csv")
    print(f"Wrote {len(stream_rows)} stream queries -> data/sample_stream.csv")


if __name__ == "__main__":
    main()
