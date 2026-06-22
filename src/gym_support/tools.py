"""Tools for the gym support agents.

Two kinds live here:

* **Business tools** (`check_credits`, `cancel_membership`, `list_classes`,
  `book_class`) — the things specialists actually do. They hit the mock data
  store and return plain strings.
* **Handoff tools** (`transfer_to_*`) — the mechanism that moves control
  between agent nodes. Each returns a ``Command`` that updates ``active_agent``
  and jumps to another node in the *parent* graph via ``graph=Command.PARENT``.

The handoff tools follow the LangChain "multiple agent subgraphs" pattern: they
forward the AIMessage that triggered the handoff plus a ToolMessage closing the
tool call, so the receiving agent sees a valid, minimal conversation history.
"""

from __future__ import annotations

from langchain.messages import AIMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from . import mock_data

# --- Business tools ---------------------------------------------------------


@tool
def check_credits(membership_id: str) -> str:
    """Look up how many credits remain on a membership, by credit type."""
    member = mock_data.get_membership(membership_id)
    lines = [f"- {ctype.replace('_', ' ')}: {n}" for ctype, n in member["credits"].items()]
    return (
        f"Membership {member['membership_id']} ({member['plan']}, "
        f"status: {member['status']}) has these credits left:\n" + "\n".join(lines)
    )


@tool
def cancel_membership(membership_id: str) -> str:
    """Cancel a membership. Returns a confirmation."""
    member = mock_data.cancel_membership(membership_id)
    return (
        f"Membership {member['membership_id']} is now {member['status']}. "
        "No further charges will be made. You can rejoin anytime."
    )


@tool
def list_classes() -> str:
    """List the classes available to book this week."""
    lines = [
        f"- {k['name']} ({k['time']}) — costs 1 {k['credit_type'].replace('_', ' ')} credit"
        for k in mock_data.CLASS_SCHEDULE
    ]
    return "Available classes:\n" + "\n".join(lines)


@tool
def book_class(membership_id: str, class_name: str, date: str) -> str:
    """Book a class for a membership on a given date. Spends one credit.

    `date` is the day the customer wants to attend (e.g. "2026-06-20" or
    "next Monday") — ask the customer for it before calling this tool.
    """
    klass = mock_data.find_class(class_name)
    if klass is None:
        return (
            f"Couldn't find a class matching '{class_name}'. "
            "Use list_classes to see what's available."
        )
    result = mock_data.book_class(membership_id, klass, date)
    if not result["ok"]:
        return f"Couldn't book {klass['name']}: {result['reason']}."
    member = result["member"]
    remaining = member["credits"][klass["credit_type"]]
    return (
        f"Booked {klass['name']} ({klass['time']}) for {date}. "
        f"You have {remaining} {klass['credit_type'].replace('_', ' ')} credit(s) left."
    )


# --- Handoff tools ----------------------------------------------------------


def _handoff(goto: str, runtime: ToolRuntime, note: str) -> Command:
    """Build a Command that transfers control to another agent node.

    Passes only the triggering AIMessage and a closing ToolMessage to the
    parent graph — not the whole sub-agent history — so the receiving agent
    gets valid, focused context.
    """
    last_ai = next(
        msg for msg in reversed(runtime.state["messages"]) if isinstance(msg, AIMessage)
    )
    transfer_message = ToolMessage(content=note, tool_call_id=runtime.tool_call_id)
    return Command(
        goto=goto,
        update={"active_agent": goto, "messages": [last_ai, transfer_message]},
        graph=Command.PARENT,
    )


@tool
def transfer_to_cancellation(runtime: ToolRuntime) -> Command:
    """Transfer the customer to the membership cancellation specialist."""
    return _handoff("cancellation", runtime, "Transferred to cancellation specialist.")


@tool
def transfer_to_credits(runtime: ToolRuntime) -> Command:
    """Transfer the customer to the credits specialist."""
    return _handoff("credits", runtime, "Transferred to credits specialist.")


@tool
def transfer_to_booking(runtime: ToolRuntime) -> Command:
    """Transfer the customer to the class booking specialist."""
    return _handoff("booking", runtime, "Transferred to booking specialist.")


@tool
def transfer_to_triage(runtime: ToolRuntime) -> Command:
    """Hand the customer back to the front desk to be re-routed."""
    return _handoff("triage", runtime, "Transferred back to the front desk.")
