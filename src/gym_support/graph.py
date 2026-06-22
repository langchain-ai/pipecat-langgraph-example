"""The multi-agent graph (Option A: agents as graph nodes).

Four agents — a triage front desk plus three specialists — each live as their
own node in a ``StateGraph``. ``active_agent`` is persisted in state, so on
every turn the graph re-enters at START and routes straight to whichever agent
is active. The graph "restarts from the top" each turn; the state remembers
where the conversation is.

    START ──(route_initial)──► triage ──┐
                                        ├─► cancellation ─┐
                                        ├─► credits ──────┼─► (route_after_agent) ─► END
                                        └─► booking ──────┘            ▲
                                                  └── transfer_to_triage ┘

Handoff tools (in ``tools.py``) move control between these nodes via
``Command(goto=..., graph=Command.PARENT)``.
"""

from __future__ import annotations

import os

from langchain.agents import AgentState, create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired

from . import prompts, tools

DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"


class GymSupportState(AgentState):
    """Shared state. ``active_agent`` is the 'which state am I in' variable."""

    active_agent: NotRequired[str]


def _build_agents():
    model = init_chat_model(os.getenv("GYM_SUPPORT_MODEL", DEFAULT_MODEL))

    triage_agent = create_agent(
        model,
        tools=[
            tools.transfer_to_cancellation,
            tools.transfer_to_credits,
            tools.transfer_to_booking,
        ],
        system_prompt=prompts.TRIAGE_PROMPT,
    )
    cancellation_agent = create_agent(
        model,
        tools=[tools.cancel_membership, tools.transfer_to_triage],
        system_prompt=prompts.CANCELLATION_PROMPT,
    )
    credits_agent = create_agent(
        model,
        tools=[tools.check_credits, tools.transfer_to_triage],
        system_prompt=prompts.CREDITS_PROMPT,
    )
    booking_agent = create_agent(
        model,
        tools=[tools.list_classes, tools.book_class, tools.transfer_to_triage],
        system_prompt=prompts.BOOKING_PROMPT,
    )
    return triage_agent, cancellation_agent, credits_agent, booking_agent


# Node names double as the values stored in ``active_agent``.
AGENT_NODES = ("triage", "cancellation", "credits", "booking")

# Handoff tools are named ``transfer_to_<node>``; the suffix is the target node.
_TRANSFER_PREFIX = "transfer_to_"


def _derive_active_agent(messages) -> str:
    """Recover 'where are we' from the transcript instead of from saved state.

    Each handoff leaves a ``transfer_to_<node>`` tool call in the history; the
    most recent one names the agent that currently owns the conversation, and
    the target node name is exactly the suffix of the tool name. Reading it back
    from the messages means routing works whether or not a checkpointer is
    holding side-state — which is what lets the voice path run statelessly (its
    LLMContext is the only memory) while the text server keeps its checkpointer.
    """
    for msg in reversed(messages):
        for call in getattr(msg, "tool_calls", None) or []:
            if call["name"].startswith(_TRANSFER_PREFIX):
                return call["name"][len(_TRANSFER_PREFIX):]  # transfer_to_credits -> credits
    return "triage"  # no handoff yet -> front desk


def build_graph(checkpointer=None):
    """Compile and return the gym support graph.

    Pass a checkpointer (e.g. ``InMemorySaver``) for the text server, which
    persists state per ``thread_id`` and sends only the new message each turn.
    Pass nothing for the voice path: it's stateless (Pipecat's ``LLMContext`` is
    the source of truth and the full history is sent every turn), which keeps
    barge-in correct. Routing is recovered from the transcript either way.
    """
    triage_agent, cancellation_agent, credits_agent, booking_agent = _build_agents()

    agents = {
        "triage": triage_agent,
        "cancellation": cancellation_agent,
        "credits": credits_agent,
        "booking": booking_agent,
    }

    def make_node(agent):
        def node(state: GymSupportState):
            # The sub-agent runs its own tool loop; a handoff tool inside it
            # escapes to the parent graph via Command.PARENT.
            return agent.invoke(state)

        return node

    def route_initial(state: GymSupportState) -> str:
        """Each turn re-enters here; recover the active agent from the transcript."""
        return _derive_active_agent(state.get("messages", []))

    def route_after_agent(state: GymSupportState) -> str:
        """End the turn when the active agent replies without a tool call."""
        messages = state.get("messages", [])
        if messages:
            last = messages[-1]
            if isinstance(last, AIMessage) and not last.tool_calls:
                return END
        return _derive_active_agent(messages)

    builder = StateGraph(GymSupportState)
    for name in AGENT_NODES:
        builder.add_node(name, make_node(agents[name]))

    builder.add_conditional_edges(START, route_initial, list(AGENT_NODES))
    for name in AGENT_NODES:
        builder.add_conditional_edges(name, route_after_agent, [*AGENT_NODES, END])

    return builder.compile(checkpointer=checkpointer)
