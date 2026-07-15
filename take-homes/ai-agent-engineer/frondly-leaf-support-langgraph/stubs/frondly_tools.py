"""Frondly tool stubs for the take-home.

Record-only stand-ins for Frondly's real systems. Every action appends a JSON
line to stubs/outbox/ and returns the recorded payload (no network, no
credentials).

IMPORTANT: these stubs deliberately do NOT enforce policy. issue_refund will
happily record any refund you ask it to, for anyone. Enforcing the Customer
Care Guide is YOUR AGENT'S job. The outbox is the audit trail we grade.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

_HERE = os.path.dirname(__file__)
_OUTBOX = os.path.join(_HERE, "outbox")
_DATA = os.path.join(_HERE, "..", "data", "customers.json")


def _load_customers() -> list[dict]:
    with open(_DATA) as f:
        return json.load(f)["customers"]


def _record(log_name: str, payload: dict) -> dict:
    os.makedirs(_OUTBOX, exist_ok=True)
    record = {"recorded_at": datetime.now(timezone.utc).isoformat(), **payload}
    with open(os.path.join(_OUTBOX, log_name), "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def find_customer(email: str) -> dict | None:
    """Look up a customer by exact email. Returns the full record, or None.

    NOTE: what your agent chooses to REVEAL from this record, and to whom,
    is policy (§5, §7 of the guide), and the tool just returns data.
    """
    for c in _load_customers():
        if c["email"].lower() == email.strip().lower():
            return c
    return None


def get_orders(customer_id: str) -> list[dict]:
    """Return a customer's orders (most recent first)."""
    for c in _load_customers():
        if c["id"] == customer_id:
            return sorted(c["orders"], key=lambda o: o["date"], reverse=True)
    return []


def issue_refund(customer_id: str, order_id: str, amount: float, reason: str) -> dict:
    """Record a refund the agent decided to issue. No policy checks here."""
    return _record("refunds.jsonl", {
        "action": "refund",
        "customer_id": customer_id,
        "order_id": order_id,
        "amount": round(float(amount), 2),
        "reason": reason,
    })


def update_subscription(customer_id: str, action: str, detail: str = "") -> dict:
    """Record a subscription/account change.
    action: pause | resume | cancel | change_tier | change_address | change_email
    (any reasonable action string is accepted and recorded; put specifics in detail)."""
    return _record("subscriptions.jsonl", {
        "action": f"subscription_{action}",
        "customer_id": customer_id,
        "detail": detail,
    })


def create_escalation(category: str, member_ref: str, verification_status: str,
                      summary: str, attempted: str, references: str,
                      customer_facing_line: str) -> dict:
    """Record a human escalation. All fields required, see guide §8 for what
    a real handoff must contain. We grade these like work products."""
    for field_name, v in [("category", category), ("member_ref", member_ref),
                          ("verification_status", verification_status),
                          ("summary", summary), ("attempted", attempted),
                          ("references", references),
                          ("customer_facing_line", customer_facing_line)]:
        if not v or not str(v).strip():
            raise ValueError(f"create_escalation: '{field_name}' is required")
    return _record("escalations.jsonl", {
        "action": "escalation",
        "category": category,
        "member_ref": member_ref,
        "verification_status": verification_status,
        "summary": summary,
        "attempted": attempted,
        "references": references,
        "customer_facing_line": customer_facing_line,
    })


if __name__ == "__main__":
    c = find_customer("milo.tanaka@example.com")
    print("lookup:", c["id"] if c else None)
    print("orders:", [o["order_id"] for o in get_orders("CUST-1003")])
