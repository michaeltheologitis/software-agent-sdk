"""ACP sub-agent sessions: routing, merging and persisting an agent's children.

Units feed the bridge (``_OpenHandsACPBridge(subagents=True)``) wire updates, the
way the ACP connection does, and read what it hands the turn and the
conversation's emitter. The rest run a real ``LocalConversation`` against the
scripted ACP agent in ``tests/fixtures/acp/scripted_agent.py``: its
``--subagents`` run, or a JSONL transcript it replays.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from acp.client.connection import ClientSideConnection
from acp.schema import ClientCapabilities, InitializeRequest, SessionNotification
from acp.utils import serialize_params

from openhands.sdk.agent.acp_agent import (
    ACPAgent,
    _fingerprint_session_id,
    _OpenHandsACPBridge,
)
from openhands.sdk.agent.acp_subagents import (
    ACPSessionNotCancellableError,
    ACPSessionNotFoundError,
)
from openhands.sdk.agent.acp_unstable import (
    SessionMessage,
    SessionMessageChunk,
    SubagentUpdate,
)
from openhands.sdk.conversation import LocalConversation
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import (
    ACPSessionMessageEvent,
    ACPSessionTextEvent,
    ACPSubagentEvent,
    ACPToolCallEvent,
    Event,
)
from tests.conftest import scripted_acp_command, subagent_snapshots, wait_until
from tests.fixtures.acp.scripted_agent import (
    announce,
    idle,
    message,
    message_chunk,
    said,
    subagent,
    text,
    thought,
    tool_call,
    tool_done,
    usage,
)


ROOT = "root"
UNSTABLE_MODELS: dict[str, Any] = {
    "subagent_update": SubagentUpdate,
    "session_message": SessionMessage,
    "session_message_chunk": SessionMessageChunk,
}


# -- The bridge, fed directly ------------------------------------------------------


class Wire:
    """A bridge with the opt-in on: the turn's events and the emitter's events."""

    def __init__(self) -> None:
        self.bridge = _OpenHandsACPBridge(subagents=True)
        assert self.bridge.subagents is not None
        self.sessions = self.bridge.subagents
        self.sessions.root_session_id = ROOT
        self.turn: list[Event] = []
        self.emitted: list[Event] = []
        self.bridge.on_event = self.turn.append
        self.bridge.on_session_event = self.emitted.append

    async def send(self, session_id: str, *updates: dict[str, Any]) -> None:
        for update in updates:
            model = UNSTABLE_MODELS.get(update["sessionUpdate"])
            if model is not None:
                parsed = model.model_validate(update)
                self.bridge.unstable_session_update(session_id, parsed)
                continue
            notification = SessionNotification.model_validate(
                {"sessionId": session_id, "update": update}
            )
            await self.bridge.session_update(session_id, notification.update)

    def latest(self, child: str) -> ACPSubagentEvent:
        return subagent_snapshots(self.emitted)[child]

    def agent(self) -> ACPAgent:
        """An agent bound to this bridge, with no process behind it."""
        agent = ACPAgent(acp_command=["unused"], acp_subagents=True)
        agent._client = self.bridge
        agent._session_id = ROOT
        return agent


def kinds(events: Sequence[Event]) -> list[str]:
    return [type(e).__name__ for e in events]


@pytest.fixture
async def wire() -> Wire:
    return Wire()


async def test_announcement_stores_parent_cell_and_cancel_grant(wire):
    await wire.send(ROOT, tool_call("cell-1"))
    await wire.send(ROOT, announce("child-a", title="Part A"))

    snapshot = wire.latest("child-a")
    assert snapshot.model_dump(
        include={
            "parent_session_id",
            "parent_tool_call_id",
            "title",
            "state",
            "cancellable",
            "meta",
        }
    ) == {
        "parent_session_id": None,
        "parent_tool_call_id": "cell-1",
        "title": "Part A",
        "state": "running",
        "cancellable": True,
        "meta": {"openhands": {"parentToolCallId": "cell-1"}},
    }
    assert kinds(wire.turn) == ["ACPToolCallEvent"]
    wire.sessions.check_cancel("child-a")


@pytest.mark.parametrize(
    "patch, changed",
    [
        ({}, {}),
        ({"title": None}, {"title": None}),
        ({"title": "New"}, {"title": "New"}),
        ({"description": "About"}, {"description": "About"}),
        ({"state": None}, {"state": None, "stop_reason": None}),
        (
            {"state": {"state": "idle", "stopReason": "end_turn"}},
            {"state": "idle", "stop_reason": "end_turn"},
        ),
        ({"state": {"state": "_reviewing"}}, {"state": "_reviewing"}),
        ({"capabilities": None}, {"cancellable": False}),
        ({"capabilities": {}}, {"cancellable": False}),
        ({"_meta": {"x": 1}}, {"meta": {"x": 1}}),
    ],
    ids=[
        "nothing",
        "title-null",
        "title-value",
        "description-value",
        "state-null",
        "state-value",
        "state-custom",
        "capabilities-null",
        "capabilities-without-cancel",
        "meta-value",
    ],
)
async def test_omitted_field_keeps_value_and_null_clears_it(wire, patch, changed):
    await wire.send(ROOT, announce("child-a", title="Part A"))
    before = wire.latest("child-a").model_dump(exclude={"id", "timestamp"})

    await wire.send(ROOT, subagent("child-a", **patch))

    after = wire.latest("child-a").model_dump(exclude={"id", "timestamp"})
    assert after == {**before, **changed}


async def test_parent_tool_call_id_survives_meta_without_it(wire):
    await wire.send(ROOT, announce("child-a", cell="cell-1"))

    await wire.send(ROOT, subagent("child-a", _meta={"x": 1}))
    assert wire.latest("child-a").parent_tool_call_id == "cell-1"

    await wire.send(ROOT, subagent("child-a", _meta=None))
    assert wire.latest("child-a").parent_tool_call_id == "cell-1"
    assert wire.latest("child-a").meta is None

    await wire.send(
        ROOT, subagent("child-a", _meta={"openhands": {"parentToolCallId": "cell-2"}})
    )
    assert wire.latest("child-a").parent_tool_call_id == "cell-2"


async def test_child_is_never_reparented_nor_its_own_parent(wire, caplog):
    await wire.send(ROOT, announce("child-a"), announce("child-c"))

    with caplog.at_level(logging.WARNING):
        await wire.send("child-c", subagent("child-a", title="Moved?"))
        await wire.send("child-b", announce("child-b"))
        await wire.send("child-a", announce(ROOT))

    assert wire.latest("child-a").parent_session_id is None
    assert wire.latest("child-a").title == "Moved?"
    assert not wire.sessions.is_child("child-b")
    assert not wire.sessions.is_child(ROOT)
    refusals = [r for r in caplog.records if r.name.endswith(".acp_subagents")]
    assert len(refusals) == 3


async def test_child_text_never_reaches_the_root_answer(wire):
    await wire.send(ROOT, announce("child-a"))

    await wire.send("child-a", said("child words"), thought("child thoughts"))
    await wire.send(ROOT, said("root words"), thought("root thoughts"))

    assert wire.bridge.accumulated_text == ["root words"]
    assert wire.bridge.accumulated_thoughts == ["root thoughts"]


async def test_child_text_is_stored_per_segment_in_transcript_order(wire):
    await wire.send(ROOT, announce("child-a"))
    wire.emitted.clear()

    await wire.send(
        "child-a",
        thought("Reading "),
        thought("part A."),
        said("Found it."),
        thought("Checking."),
        tool_call("cell-a1"),
    )

    assert [
        (e.thought, e.text) for e in wire.emitted if isinstance(e, ACPSessionTextEvent)
    ] == [(True, "Reading part A."), (False, "Found it."), (True, "Checking.")]
    assert kinds(wire.emitted) == [
        "ACPSessionTextEvent",
        "ACPSessionTextEvent",
        "ACPSessionTextEvent",
        "ACPToolCallEvent",
    ]


async def test_usage_never_splits_a_text_segment(wire):
    await wire.send(ROOT, announce("child-a"))

    await wire.send("child-a", thought("one "), usage(0.0001), thought("segment"))
    await wire.send(ROOT, idle("child-a"))

    texts = [e.text for e in wire.emitted if isinstance(e, ACPSessionTextEvent)]
    assert texts == ["one segment"]


async def test_chunked_message_is_stored_whole_at_the_next_boundary(wire):
    await wire.send(ROOT, announce("child-a"))
    await wire.send("child-a", announce("child-a-1", cell=None))
    wire.emitted.clear()

    await wire.send(
        "child-a-1",
        message_chunk("answer", "child-a-1", "child-a", "Part A "),
        message_chunk("answer", "child-a-1", "child-a", "checks out."),
    )
    assert wire.emitted == []

    await wire.send("child-a", idle("child-a-1"))

    stored, snapshot = wire.emitted
    assert isinstance(stored, ACPSessionMessageEvent)
    assert (stored.acp_session_id, stored.message_id, stored.text) == (
        "child-a-1",
        "answer",
        "Part A checks out.",
    )
    assert (stored.sender_session_id, stored.recipient_session_id) == (
        "child-a-1",
        "child-a",
    )
    assert isinstance(snapshot, ACPSubagentEvent)
    assert snapshot.state == "idle"


async def test_message_upsert_replaces_content_and_keeps_participants(wire):
    await wire.send(ROOT, announce("child-a"))
    await wire.send(ROOT, message("task", ROOT, "child-a", "First."))

    await wire.send(
        ROOT,
        {
            "sessionUpdate": "session_message",
            "messageId": "task",
            "content": [text("Second.")],
        },
    )
    await wire.send(
        ROOT,
        {"sessionUpdate": "session_message", "messageId": "task", "content": None},
    )

    stored = [e for e in wire.emitted if isinstance(e, ACPSessionMessageEvent)]
    assert [(e.acp_session_id, e.sender_session_id, e.text) for e in stored] == [
        (None, ROOT, "First."),
        (None, ROOT, "Second."),
        (None, ROOT, ""),
    ]
    assert {e.recipient_session_id for e in stored} == {"child-a"}


async def test_child_cost_is_on_its_association_and_never_booked_to_the_conversation(
    wire,
):
    agent = wire.agent()
    await wire.send(ROOT, announce("child-a"))

    await wire.send("child-a", usage(0.0004))
    await wire.send("child-a", usage(0.0004))
    await wire.send(ROOT, usage(0.0011))
    agent._record_usage(
        None, ROOT, usage_update=wire.bridge.pop_turn_usage_update(ROOT)
    )

    costs = [
        (e.cost, e.cost_currency)
        for e in wire.emitted
        if isinstance(e, ACPSubagentEvent) and e.acp_session_id == "child-a"
    ]
    assert costs == [(None, None), (0.0004, "USD")]
    assert agent.llm.metrics.accumulated_cost == pytest.approx(0.0011)


async def test_child_usage_leaves_root_usage_sync_and_context_window_alone(wire):
    root_usage = wire.bridge.prepare_usage_sync(ROOT)
    await wire.send(ROOT, usage(size=1000))
    await wire.send(ROOT, announce("child-a"))
    root_usage.clear()

    await wire.send("child-a", usage(0.0004, size=99))

    assert not root_usage.is_set()
    assert wire.bridge._context_window == 1000
    assert "child-a" not in wire.bridge._context_window_by_session
    assert wire.bridge.get_turn_usage_update("child-a") is None


async def test_child_tool_calls_are_keyed_by_session_and_tool_call_id(wire):
    await wire.send(ROOT, tool_call("t1"), announce("child-a"))
    await wire.send("child-a", tool_call("t1", _meta={"cell": 1}))

    await wire.send("child-a", tool_done("t1", _meta={"cell": 2}))

    [root_call] = [e for e in wire.turn if isinstance(e, ACPToolCallEvent)]
    child_calls = [e for e in wire.emitted if isinstance(e, ACPToolCallEvent)]
    assert (root_call.acp_session_id, root_call.status) == (None, "in_progress")
    assert [(e.acp_session_id, e.status, e.meta) for e in child_calls] == [
        ("child-a", "in_progress", {"cell": 1}),
        ("child-a", "completed", {"cell": 2}),
    ]


async def test_turn_end_force_completes_only_root_tool_calls(wire):
    agent = wire.agent()
    await wire.send(ROOT, tool_call("r1"), announce("child-a"))
    await wire.send("child-a", tool_call("c1"))
    wire.turn.clear()
    wire.emitted.clear()

    agent._flush_inflight_tool_calls_as_completed()

    assert [(e.tool_call_id, e.status) for e in wire.turn] == [("r1", "completed")]
    assert wire.emitted == []


async def test_aborted_turn_fails_child_tool_calls_with_their_session(wire):
    agent = wire.agent()
    await wire.send(ROOT, tool_call("r1"), announce("child-a"))
    await wire.send("child-a", tool_call("c1", _meta={"cell": 1}))
    wire.turn.clear()
    wire.emitted.clear()

    agent._cancel_inflight_tool_calls()

    assert [(e.tool_call_id, e.status, e.acp_session_id) for e in wire.turn] == [
        ("r1", "failed", None)
    ]
    assert [
        (e.tool_call_id, e.status, e.acp_session_id, e.meta)
        for e in wire.emitted
        if isinstance(e, ACPToolCallEvent)
    ] == [("c1", "failed", "child-a", {"cell": 1})]


async def test_child_call_open_across_aborted_turns_is_failed_once_and_the_agents_report_wins(  # noqa: E501
    wire,
):
    agent = wire.agent()
    await wire.send(ROOT, announce("child-a"))
    await wire.send("child-a", tool_call("c1"))

    for _aborted_turn in range(2):
        agent._cancel_inflight_tool_calls()
        wire.bridge.reset()
    await wire.send("child-a", tool_done("c1"))

    assert [
        (e.tool_call_id, e.status)
        for e in wire.emitted
        if isinstance(e, ACPToolCallEvent)
    ] == [("c1", "in_progress"), ("c1", "failed"), ("c1", "completed")]


async def test_child_events_go_to_the_session_emitter_and_root_events_to_the_turn(
    wire,
):
    await wire.send(ROOT, tool_call("cell-1"), announce("child-a"))
    await wire.send(ROOT, message("task", ROOT, "child-a", "Go."))
    await wire.send("child-a", thought("Hm."), tool_call("c1"), tool_done("c1"))

    assert kinds(wire.turn) == ["ACPToolCallEvent"]
    assert kinds(wire.emitted) == [
        "ACPSubagentEvent",
        "ACPSessionMessageEvent",
        "ACPSessionTextEvent",
        "ACPToolCallEvent",
        "ACPToolCallEvent",
    ]


async def test_child_traffic_between_turns_reaches_the_emitter_in_order(wire):
    await wire.send(ROOT, announce("child-a"))
    wire.bridge.reset()
    wire.emitted.clear()

    await wire.send(ROOT, tool_call("late-root"))
    await wire.send("child-a", tool_call("c1"), tool_done("c1"))
    await wire.send(ROOT, idle("child-a"))

    assert wire.turn == []
    assert [
        (type(e).__name__, getattr(e, "status", None) or getattr(e, "state", None))
        for e in wire.emitted
    ] == [
        ("ACPToolCallEvent", "in_progress"),
        ("ACPToolCallEvent", "completed"),
        ("ACPSubagentEvent", "idle"),
    ]


async def test_child_events_without_an_emitter_are_dropped_with_a_debug_line(
    wire, caplog
):
    wire.bridge.on_session_event = None

    with caplog.at_level(logging.DEBUG, logger="openhands.sdk.agent.acp_agent"):
        await wire.send(ROOT, announce("child-a"))

    assert wire.sessions.is_child("child-a")
    assert any("no emitter" in r.getMessage() for r in caplog.records)


async def test_replay_is_neither_stored_nor_grants_cancel(wire):
    with wire.bridge.replaying(ROOT):
        await wire.send(ROOT, announce("child-a"))
        await wire.send("child-a", said("old words"), tool_call("c1"))
        await wire.send(ROOT, message("task", ROOT, "child-a", "Go."))

    assert wire.emitted == []
    assert wire.bridge.accumulated_text == []
    assert wire.sessions.is_child("child-a")
    with pytest.raises(ACPSessionNotCancellableError):
        wire.sessions.check_cancel("child-a")


async def test_replayed_child_calls_are_never_tracked_nor_failed_later(wire):
    agent = wire.agent()
    with wire.bridge.replaying(ROOT):
        await wire.send(ROOT, announce("child-a"))
        await wire.send("child-a", tool_call("c1"))

    wire.bridge.reset()
    agent._cancel_inflight_tool_calls()
    await wire.send("child-a", tool_done("c1"))

    assert wire.bridge.accumulated_tool_calls == []
    assert [e for e in wire.emitted if isinstance(e, ACPToolCallEvent)] == []


def stored_snapshot(child: str, **fields: Any) -> ACPSubagentEvent:
    return ACPSubagentEvent(
        parent_id=f"event-before-{child}",
        acp_session_id=child,
        title=f"Title of {child}",
        parent_tool_call_id="cell-1",
        cost=0.5,
        cost_currency="USD",
        meta={"k": child},
        **fields,
    )


async def test_new_connection_withdraws_cancel_and_unconfirms_state(wire):
    history = [
        stored_snapshot("active", state="running", cancellable=True),
        stored_snapshot("finished", state="idle", cancellable=True),
        stored_snapshot("unconfirmed", state=None),
        stored_snapshot("custom", state="_reviewing"),
    ]

    reset = wire.sessions.seed(history)

    assert [e.acp_session_id for e in reset] == ["active", "custom"]
    for event, stored in zip(reset, (history[0], history[3])):
        assert event.id != stored.id
        assert event.model_dump(exclude={"id", "timestamp"}) == {
            **stored.model_dump(exclude={"id", "timestamp"}),
            "source": "environment",
            "parent_id": None,
            "state": None,
            "stop_reason": None,
            "cancellable": False,
        }
    for child in ("active", "finished", "unconfirmed", "custom"):
        assert wire.sessions.is_child(child)
        with pytest.raises(ACPSessionNotCancellableError):
            wire.sessions.check_cancel(child)


async def test_partial_patch_after_reconnect_keeps_the_stored_title(wire):
    wire.sessions.seed([stored_snapshot("child-a", state="running")])

    await wire.send(ROOT, idle("child-a"))

    snapshot = wire.latest("child-a")
    assert (snapshot.title, snapshot.parent_tool_call_id, snapshot.cost) == (
        "Title of child-a",
        "cell-1",
        0.5,
    )
    assert (snapshot.state, snapshot.parent_id) == ("idle", None)


async def test_unannounced_session_follows_the_root_path_with_one_warning(wire, caplog):
    with caplog.at_level(logging.WARNING):
        await wire.send("stranger", said("one "), said("two"), tool_call("s1"))

    assert wire.bridge.accumulated_text == ["one ", "two"]
    assert [e.acp_session_id for e in wire.turn] == [None]
    assert len(caplog.records) == 1
    assert "never announced" in caplog.records[0].getMessage()


async def test_unstable_updates_on_an_unannounced_session_stay_under_that_session(
    wire, caplog
):
    stranger = "stranger-session"
    with caplog.at_level(logging.WARNING):
        await wire.send(
            stranger,
            announce("child-x"),
            message("task", stranger, "child-x", "Go."),
        )

    assert wire.latest("child-x").parent_session_id == stranger
    [stored_message] = [
        e for e in wire.emitted if isinstance(e, ACPSessionMessageEvent)
    ]
    assert stored_message.acp_session_id == stranger
    assert (wire.bridge.accumulated_text, wire.bridge.accumulated_tool_calls) == (
        [],
        [],
    )
    assert wire.turn == []
    [warning] = caplog.records
    assert "never announced" in warning.getMessage()
    assert _fingerprint_session_id(stranger) in warning.getMessage()


# -- Through a conversation, against the scripted agent ---------------------------


def turned_idle(conv: LocalConversation, child: str) -> ACPSubagentEvent | None:
    snapshot = subagent_snapshots(conv.state.events).get(child)
    return snapshot if snapshot is not None and snapshot.state == "idle" else None


def tree_from(events: Sequence[Event]) -> dict[str | None, dict[str, Any]]:
    """The sub-agent tree stored ``events`` describe, read by the persisted
    contract: latest snapshot per child, placed by its spawning cell;
    per-session tool calls and messages, last wins."""
    snapshots = subagent_snapshots(events)
    calls: dict[tuple[str | None, str], ACPToolCallEvent] = {}
    messages: dict[tuple[str | None, str], ACPSessionMessageEvent] = {}
    texts: dict[str, list[tuple[bool, str]]] = {}
    for event in events:
        if isinstance(event, ACPToolCallEvent):
            calls[(event.acp_session_id, event.tool_call_id)] = event
        elif isinstance(event, ACPSessionMessageEvent):
            messages[(event.acp_session_id, event.message_id)] = event
        elif isinstance(event, ACPSessionTextEvent):
            texts.setdefault(event.acp_session_id, []).append(
                (event.thought, event.text)
            )

    def placement(child: str) -> tuple[Any, ...]:
        snapshot = snapshots[child]
        cell = (snapshot.parent_session_id, snapshot.parent_tool_call_id or "")
        return (
            ("cell", snapshot.parent_tool_call_id) if cell in calls else ("unplaced",)
        )

    tree: dict[str | None, dict[str, Any]] = {}
    for session in [None, *snapshots]:
        node: dict[str, Any] = {
            "tool_calls": {
                call_id: call.status
                for (owner, call_id), call in calls.items()
                if owner == session
            },
            "messages": {
                message_id: (m.sender_session_id, m.recipient_session_id, m.text)
                for (owner, message_id), m in messages.items()
                if owner == session
            },
        }
        if session is not None:
            snapshot = snapshots[session]
            node |= {
                "parent": snapshot.parent_session_id,
                "placement": placement(session),
                "state": (snapshot.state, snapshot.stop_reason),
                "cancellable": snapshot.cancellable,
                "cost": (snapshot.cost, snapshot.cost_currency),
                "texts": texts.get(session, []),
            }
        tree[session] = node
    return tree


def child_node(parent: str | None, cell: str, **fields: Any) -> dict[str, Any]:
    return {
        "parent": parent,
        "placement": ("cell", cell),
        "state": ("idle", "end_turn"),
        "cancellable": True,
        "cost": (None, None),
        "tool_calls": {},
        "messages": {},
        "texts": [],
        **fields,
    }


def scripted_tree(root: str) -> dict[str | None, dict[str, Any]]:
    """What the scripted agent's ``--subagents`` run stores, without a cancel."""
    return {
        None: {
            "tool_calls": {"cell-1": "completed"},
            "messages": {"child-a-task": (root, "child-a", "Summarize part A.")},
        },
        "child-a": child_node(
            None,
            "cell-1",
            cost=(0.0004, "USD"),
            tool_calls={"cell-a1": "completed"},
            messages={"child-a-answer": ("child-a", root, "Part A: fine.")},
            texts=[(True, "Reading part A.")],
        ),
        "child-a-1": child_node(
            "child-a",
            "cell-a1",
            messages={
                "child-a-1-answer": ("child-a-1", "child-a", "Part A checks out.")
            },
        ),
        "child-c": child_node(
            None, "cell-1", cancellable=False, tool_calls={"cell-c1": "completed"}
        ),
        "child-b": child_node(None, "cell-1", tool_calls={"cell-b1": "completed"}),
    }


@pytest.fixture
def conversation(scripted_conversation) -> Callable[..., LocalConversation]:
    """``scripted_conversation`` with sub-agent sessions on."""
    return partial(scripted_conversation, acp_subagents=True)


def run(conv: LocalConversation, prompt: str = "hello") -> list[Event]:
    conv.send_message(prompt)
    conv.run()
    return list(conv.state.events)


def root_id(conv: LocalConversation) -> str:
    return conv.state.agent_state["acp_session_id"]


def test_scripted_run_stores_the_scripted_tree(conversation):
    conv = conversation("--subagents")

    run(conv)
    wait_until(lambda: turned_idle(conv, "child-b"))

    assert tree_from(list(conv.state.events)) == scripted_tree(root_id(conv))


def test_scripted_run_books_only_the_roots_cost(conversation):
    conv = conversation("--subagents")

    run(conv)
    wait_until(lambda: turned_idle(conv, "child-b"))

    assert conv.state.stats.get_combined_metrics().accumulated_cost == pytest.approx(
        0.0011
    )


def test_subagents_off_stores_only_root_work_through_the_stock_connection(
    conversation, acp_request_log
):
    conv = conversation("--subagents", acp_subagents=False)

    events = run(conv)

    assert type(conv.agent._conn) is ClientSideConnection
    [initialize] = [e for e in acp_request_log() if e["method"] == "initialize"]
    library_call = InitializeRequest(
        protocol_version=1, client_capabilities=ClientCapabilities()
    )
    assert initialize["params"] == serialize_params(library_call)
    calls = [e for e in events if isinstance(e, ACPToolCallEvent)]
    assert {e.tool_call_id for e in calls} == {"cell-1"}
    assert all("acp_session_id" not in e.model_dump(exclude_none=True) for e in calls)
    new_kinds = (ACPSubagentEvent, ACPSessionMessageEvent, ACPSessionTextEvent)
    assert not [e for e in events if isinstance(e, new_kinds)]


def cancel_once_announced(conv: LocalConversation, child: str) -> bool:
    """Retry until the child is known on the live connection; return whether
    another thread held the conversation's state lock when the call that
    succeeded began."""
    deadline = time.monotonic() + 15
    while True:
        lock_held_elsewhere = conv.state._lock.locked() and not (
            conv.state._lock.owned()
        )
        try:
            conv.cancel_acp_session(child)
            return lock_held_elsewhere
        except (ACPSessionNotFoundError, ACPSessionNotCancellableError):
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)


def run_in_thread(conv: LocalConversation) -> threading.Thread:
    conv.send_message("hello")
    runner = threading.Thread(target=conv.run, daemon=True)
    runner.start()
    return runner


def test_cancel_acp_session_reaches_the_child_without_waiting_for_the_state_lock(
    conversation, acp_request_log
):
    conv = conversation("--subagents", "--cancel-wait", "30")
    runner = run_in_thread(conv)

    lock_held_by_the_run = cancel_once_announced(conv, "child-b")
    runner.join(timeout=30)

    assert lock_held_by_the_run
    assert conv.state.execution_status == ConversationExecutionStatus.FINISHED
    assert {"method": "session/cancel", "params": {"sessionId": "child-b"}} in (
        acp_request_log()
    )
    snapshot = wait_until(lambda: turned_idle(conv, "child-b"))
    assert snapshot.stop_reason == "cancelled"
    tree = tree_from(list(conv.state.events))
    assert tree["child-b"]["tool_calls"] == {"cell-b1": "failed"}


def test_cancel_acp_session_for_an_idle_child_that_keeps_its_grant_is_sent(
    conversation, acp_request_log
):
    conv = conversation("--subagents")
    run(conv)
    idle_with_grant = wait_until(lambda: turned_idle(conv, "child-b"))
    assert idle_with_grant.cancellable

    conv.cancel_acp_session("child-b")

    cancel = {"method": "session/cancel", "params": {"sessionId": "child-b"}}
    wait_until(lambda: cancel in acp_request_log())
    assert subagent_snapshots(conv.state.events)["child-b"] == idle_with_grant


def test_cancel_acp_session_refuses_a_child_without_a_grant(
    conversation, acp_request_log
):
    conv = conversation("--subagents")
    run(conv)

    with pytest.raises(ACPSessionNotCancellableError):
        conv.cancel_acp_session("child-c")

    assert "session/cancel" not in [e["method"] for e in acp_request_log()]


def test_cancel_acp_session_refuses_unknown_and_root_sessions(conversation):
    conv = conversation("--subagents")
    run(conv)

    with pytest.raises(ACPSessionNotFoundError):
        conv.cancel_acp_session("no-such-child")
    with pytest.raises(ACPSessionNotCancellableError):
        conv.cancel_acp_session(root_id(conv))


def test_cancel_acp_session_without_a_live_connection_is_refused(conversation):
    conv = conversation("--subagents")

    with pytest.raises(ACPSessionNotCancellableError):
        conv.cancel_acp_session("child-b")


# -- Transcripts --------------------------------------------------------------------


TRANSCRIPT_ROOT = "s-root"


def request(request_id: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}


def response(request_id: int, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def update(session_id: str, value: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {"sessionId": session_id, "update": value},
    }


def recorded_run(*, worker_finishes: bool = True) -> list[dict[str, Any]]:
    """A full recording of one turn: the client's requests and the agent's lines."""
    root = TRANSCRIPT_ROOT
    worker_end = (
        [update("worker", tool_done("w1")), update(root, idle("worker"))]
        if worker_finishes
        else []
    )
    return [
        request(0, "initialize", {"protocolVersion": 1}),
        response(0, {"protocolVersion": 1, "agentCapabilities": {}}),
        request(1, "session/new", {"cwd": "/w", "mcpServers": []}),
        response(1, {"sessionId": root}),
        request(2, "session/prompt", {"sessionId": root, "prompt": [text("go")]}),
        update(root, tool_call("cell-1")),
        update(root, announce("worker", title="Work")),
        update(root, message("task", root, "worker", "Do it.")),
        update(root, announce("helper", title="Help")),
        update("helper", said("Helped.")),
        update(root, idle("helper")),
        update("worker", thought("Thinking.")),
        update("worker", tool_call("w1")),
        update("worker", usage(0.0002)),
        update("worker", message("answer", "worker", root, "Done.")),
        *worker_end,
        update(root, tool_done("cell-1")),
        update(root, said("All done.")),
        update(root, usage(0.0005)),
        response(2, {"stopReason": "end_turn"}),
    ]


def outgoing_only(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What the agent sent: the shape of a recording made on the agent's side."""
    return [line for line in lines if "result" in line or "update" in line["params"]]


def write_lines(tmp_path: Path, lines: list[dict[str, Any]]) -> Path:
    path = tmp_path / f"transcript-{uuid.uuid4().hex}.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


RECORDED_TREE: dict[str | None, dict[str, Any]] = {
    None: {
        "tool_calls": {"cell-1": "completed"},
        "messages": {"task": (TRANSCRIPT_ROOT, "worker", "Do it.")},
    },
    "worker": child_node(
        None,
        "cell-1",
        cost=(0.0002, "USD"),
        tool_calls={"w1": "completed"},
        messages={"answer": ("worker", TRANSCRIPT_ROOT, "Done.")},
        texts=[(True, "Thinking.")],
    ),
    "helper": child_node(None, "cell-1", texts=[(False, "Helped.")]),
}


@pytest.mark.parametrize("recording", ["full", "outgoing-only"])
def test_scripted_transcript_replays_a_recording(conversation, tmp_path, recording):
    lines = recorded_run()
    if recording == "outgoing-only":
        lines = outgoing_only(lines)
    conv = conversation("--transcript", str(write_lines(tmp_path, lines)))

    run(conv)
    wait_until(lambda: turned_idle(conv, "worker"))

    assert tree_from(list(conv.state.events)) == RECORDED_TREE


def test_transcript_interval_paces_the_replay(conversation, tmp_path):
    lines = outgoing_only(recorded_run())
    updates = sum(1 for line in lines if "update" in line.get("params", {}))
    conv = conversation(
        "--transcript",
        str(write_lines(tmp_path, lines)),
        "--transcript-interval-ms",
        "50",
    )

    started = time.monotonic()
    run(conv)

    assert time.monotonic() - started >= updates * 0.05


def test_transcript_wait_point_that_is_never_reached_exits_non_zero(tmp_path):
    path = write_lines(tmp_path, outgoing_only(recorded_run()))
    agent = subprocess.Popen(
        scripted_acp_command("--transcript", str(path), "--wait-timeout", "0.2"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert agent.wait(timeout=10) != 0
    finally:
        agent.kill()
        for stream in (agent.stdin, agent.stdout, agent.stderr):
            assert stream is not None
            stream.close()


# -- Ordering, as the persisted contract states it ------------------------------------


def stored_at(event: Event) -> datetime:
    return datetime.fromisoformat(event.timestamp)


def session_of(event: Event) -> str | None:
    """The child an event belongs to; None for the root's and other events."""
    if isinstance(event, ACPSubagentEvent | ACPSessionTextEvent):
        return event.acp_session_id
    if isinstance(event, ACPToolCallEvent | ACPSessionMessageEvent):
        return event.acp_session_id
    return None


def test_a_childs_stored_timestamps_never_decrease_in_log_order(conversation):
    conv = conversation("--subagents")
    run(conv)
    wait_until(lambda: turned_idle(conv, "child-b"))

    by_child: dict[str, list[datetime]] = {}
    for event in conv.state.events:
        if (child := session_of(event)) is not None:
            by_child.setdefault(child, []).append(stored_at(event))

    assert set(by_child) == {"child-a", "child-a-1", "child-b", "child-c"}
    for stamps in by_child.values():
        assert stamps == sorted(stamps)


def answers(conv: LocalConversation) -> int:
    """How many times the worker's answer, its last line in a turn, is stored."""
    return sum(
        1
        for e in conv.state.events
        if isinstance(e, ACPSessionMessageEvent) and e.message_id == "answer"
    )


def test_a_reconnect_snapshot_is_later_than_the_childs_earlier_events(
    conversation, tmp_path
):
    transcript = str(write_lines(tmp_path, recorded_run(worker_finishes=False)))
    first = conversation("--transcript", transcript)
    run(first)
    wait_until(lambda: answers(first) == 1)
    first.close()

    second = conversation(conversation_id=first.id)
    run(second)
    wait_until(lambda: answers(second) == 2)
    events = list(second.state.events)

    reconnects = [
        e
        for e in events
        if isinstance(e, ACPSubagentEvent) and e.source == "environment"
    ]
    [reconnect] = reconnects
    assert (reconnect.acp_session_id, reconnect.state, reconnect.cancellable) == (
        "worker",
        None,
        False,
    )
    position = events.index(reconnect)
    earlier = [e for e in events[:position] if session_of(e) == "worker"]
    assert earlier
    assert all(stored_at(e) < stored_at(reconnect) for e in earlier)
    later = [e for e in events[position + 1 :] if session_of(e) == "worker"]
    assert later
    assert all(stored_at(reconnect) <= stored_at(e) for e in later)


def test_a_spawning_cells_started_event_precedes_its_whole_subtree(conversation):
    conv = conversation("--subagents")
    run(conv)
    wait_until(lambda: turned_idle(conv, "child-b"))
    events = list(conv.state.events)

    snapshots = subagent_snapshots(events)
    children: dict[str | None, set[str]] = {}
    for child, snapshot in snapshots.items():
        children.setdefault(snapshot.parent_session_id, set()).add(child)

    def subtree(child: str) -> set[str]:
        return {child}.union(*(subtree(c) for c in children.get(child, set())))

    assert sorted(snapshots) == ["child-a", "child-a-1", "child-b", "child-c"]
    for child, snapshot in snapshots.items():
        assert snapshot.parent_tool_call_id is not None
        started = next(
            i
            for i, e in enumerate(events)
            if isinstance(e, ACPToolCallEvent)
            and e.acp_session_id == snapshot.parent_session_id
            and e.tool_call_id == snapshot.parent_tool_call_id
        )
        members = subtree(child)
        for position, later in enumerate(events):
            if session_of(later) in members:
                assert position > started
                assert stored_at(later) >= stored_at(events[started])
