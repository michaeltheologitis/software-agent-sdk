"""The persisted kinds of ACP sub-agent sessions, and the two new tool-call fields."""

import json

import pytest

from openhands.sdk.event import (
    ACPSessionMessageEvent,
    ACPSessionTextEvent,
    ACPSubagentEvent,
    ACPToolCallEvent,
    Event,
)


SUBAGENT = ACPSubagentEvent(
    acp_session_id="child-a",
    parent_session_id=None,
    parent_tool_call_id="cell-1",
    title="Summarize part A",
    state="idle",
    stop_reason="end_turn",
    cancellable=True,
    cost=0.0004,
    cost_currency="USD",
    meta={"openhands": {"parentToolCallId": "cell-1"}, "vendor": {"node": 2}},
)
MESSAGE = ACPSessionMessageEvent(
    acp_session_id="child-a",
    message_id="answer",
    sender_session_id="child-a",
    recipient_session_id="root",
    text="Part A: fine.",
    meta={"vendor": True},
)
TEXT = ACPSessionTextEvent(acp_session_id="child-a", thought=True, text="Reading.")
CHILD_CALL = ACPToolCallEvent(
    tool_call_id="cell-a1",
    title="Run delegate",
    status="completed",
    acp_session_id="child-a",
    meta={"vendor": {"cell": 1}},
)


@pytest.mark.parametrize("event", [SUBAGENT, MESSAGE, TEXT, CHILD_CALL])
def test_subagent_events_round_trip_through_json(event: Event):
    restored = Event.model_validate_json(event.model_dump_json())

    assert type(restored) is type(event)
    assert restored == event
    assert json.loads(event.model_dump_json())["kind"] == type(event).__name__


def test_legacy_acp_tool_call_event_loads_without_session_fields():
    legacy = {
        "kind": "ACPToolCallEvent",
        "id": "e1",
        "timestamp": "2026-10-01T12:00:00",
        "source": "agent",
        "tool_call_id": "tc-1",
        "title": "Run ls",
        "status": "completed",
        "is_error": False,
    }

    restored = Event.model_validate(legacy)

    assert isinstance(restored, ACPToolCallEvent)
    assert (restored.acp_session_id, restored.meta) == (None, None)


@pytest.mark.parametrize(
    "event, shown",
    [
        (SUBAGENT, ["Summarize part A", "cell-1", "idle (end_turn)", "0.0004 USD"]),
        (MESSAGE, ["child-a -> root", "Part A: fine."]),
        (TEXT, ["child-a", "Reading."]),
    ],
)
def test_subagent_events_visualize_their_essentials(event: Event, shown: list[str]):
    rendered = event.visualize.plain

    for fragment in shown:
        assert fragment in rendered


def test_unconfirmed_state_is_shown_as_such():
    reset = SUBAGENT.model_copy(
        update={"source": "environment", "state": None, "cancellable": False}
    )

    assert "state=unconfirmed" in reset.visualize.plain
