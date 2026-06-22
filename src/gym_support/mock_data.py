"""In-memory mock data for the gym support demo.

There is no database and no external API. Every membership ID "exists": the
first time we see one, we mint a member with a standard bundle of credits. Tool
side effects (cancelling, booking) mutate this in-process dict so a single
chat session behaves consistently — restart the server and it resets.
"""

from __future__ import annotations

from copy import deepcopy

# Credits a fresh member starts with, by credit type.
_DEFAULT_CREDITS: dict[str, int] = {
    "group_class": 8,
    "personal_training": 2,
    "guest_pass": 3,
}

# The (fake) class schedule. `credit_type` is what booking a slot spends.
CLASS_SCHEDULE: list[dict[str, str]] = [
    {"id": "yoga-mon", "name": "Sunrise Yoga", "time": "Mon 7:00 AM", "credit_type": "group_class"},
    {"id": "hiit-tue", "name": "HIIT Blast", "time": "Tue 6:00 PM", "credit_type": "group_class"},
    {"id": "spin-wed", "name": "Spin Studio", "time": "Wed 5:30 PM", "credit_type": "group_class"},
    {"id": "pt-thu", "name": "1:1 Personal Training", "time": "Thu 12:00 PM", "credit_type": "personal_training"},
    {"id": "pilates-sat", "name": "Reformer Pilates", "time": "Sat 9:00 AM", "credit_type": "group_class"},
]

# membership_id -> member record. Populated lazily by get_membership().
_MEMBERS: dict[str, dict] = {}


def get_membership(membership_id: str) -> dict:
    """Return the member for this ID, minting one on first sight.

    Regardless of the ID provided, this always "finds" a membership — the demo
    has no real lookup to fail.
    """
    key = (membership_id or "guest").strip()
    if key not in _MEMBERS:
        _MEMBERS[key] = {
            "membership_id": key,
            "name": "Alex Member",
            "plan": "Unlimited Monthly",
            "status": "active",
            "credits": deepcopy(_DEFAULT_CREDITS),
            "bookings": [],
        }
    return _MEMBERS[key]


def cancel_membership(membership_id: str) -> dict:
    """Mark the membership cancelled. Idempotent."""
    member = get_membership(membership_id)
    member["status"] = "cancelled"
    return member


def find_class(class_query: str) -> dict | None:
    """Best-effort match of a class by id, name, or time substring."""
    q = (class_query or "").strip().lower()
    if not q:
        return None
    for klass in CLASS_SCHEDULE:
        if q in klass["id"].lower() or q in klass["name"].lower() or q in klass["time"].lower():
            return klass
    return None


def book_class(membership_id: str, klass: dict, date: str) -> dict:
    """Spend one credit of the class's type and record the booking.

    Returns a result dict: {"ok": bool, "reason"?: str, "member": <record>}.
    """
    member = get_membership(membership_id)
    credit_type = klass["credit_type"]
    remaining = member["credits"].get(credit_type, 0)
    if remaining <= 0:
        return {"ok": False, "reason": f"no {credit_type} credits left", "member": member}
    member["credits"][credit_type] = remaining - 1
    member["bookings"].append({"class": klass["name"], "time": klass["time"], "date": date})
    return {"ok": True, "member": member}
