"""The shim that carries ACP's unstable sub-agent types past agent-client-protocol.

The far side of every test is a real agent-side ``acp.connection.Connection``
over a socket pair, so what is asserted is what crosses the wire.
"""

from __future__ import annotations

import asyncio
import logging
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from acp.agent.router import build_agent_router
from acp.connection import Connection, JsonValue
from acp.schema import (
    AgentMessageChunk,
    ClientCapabilities,
    InitializeRequest,
    InitializeResponse,
    SessionNotification,
)
from acp.utils import serialize_params
from pydantic import ValidationError

from openhands.sdk.agent.acp_unstable import (
    SUBAGENT_CLIENT_CAPABILITIES,
    SessionMessage,
    SubagentClientSideConnection,
    SubagentUpdate,
    UnstableSessionUpdate,
)


LIBRARY_CAUGHT_UP = (
    "agent-client-protocol now parses ACP's sub-agent updates: delete "
    "openhands/sdk/agent/acp_unstable.py, use the library's types and capability, "
    "re-run the ACP conformance probes against Claude Code, Codex and Gemini, and "
    "bump agent-client-protocol in openhands-sdk/pyproject.toml."
)


def subagent_update(child: str, **fields: Any) -> dict[str, Any]:
    return {"sessionUpdate": "subagent_update", "sessionId": child, **fields}


def session_message(message_id: str, text: str) -> dict[str, Any]:
    return {
        "sessionUpdate": "session_message",
        "messageId": message_id,
        "senderSessionId": "root",
        "recipientSessionId": "child",
        "content": [{"type": "text", "text": text}],
    }


def message_chunk(text: str) -> dict[str, Any]:
    return {
        "sessionUpdate": "agent_message_chunk",
        "content": {"type": "text", "text": text},
    }


class Recorder:
    """The client side: every update, stable or not, in the order it arrived."""

    def __init__(self) -> None:
        self.received: list[tuple[str, Any]] = []

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        self.received.append((session_id, update))

    def unstable_update(self, session_id: str, update: UnstableSessionUpdate) -> None:
        self.received.append((session_id, update))


class Agent:
    """The agent side: answers initialize and remembers its raw params."""

    def __init__(self) -> None:
        self.initialize_params: dict[str, Any] = {}

    async def initialize(self, protocol_version: int, **kwargs: Any):
        return InitializeResponse(protocol_version=protocol_version)


class Wire:
    def __init__(self, client: SubagentClientSideConnection, agent: Connection) -> None:
        self.client = client
        self.agent = agent

    async def send(self, session_id: str, update: dict[str, Any]) -> None:
        await self.agent.send_notification(
            "session/update", {"sessionId": session_id, "update": update}
        )


@asynccontextmanager
async def wired(recorder: Recorder, agent: Agent | None = None) -> AsyncIterator[Wire]:
    agent = agent or Agent()
    router = build_agent_router(agent)  # pyright: ignore[reportArgumentType]

    async def tap(method: str, params: JsonValue | None, is_notification: bool):
        if method == "initialize" and isinstance(params, dict):
            agent.initialize_params.update(params)
        return await router(method, params, is_notification)

    client_socket, agent_socket = socket.socketpair()
    client_reader, client_writer = await asyncio.open_connection(sock=client_socket)
    agent_reader, agent_writer = await asyncio.open_connection(sock=agent_socket)
    client = SubagentClientSideConnection(
        recorder,  # pyright: ignore[reportArgumentType]
        client_writer,
        client_reader,
        on_unstable_update=recorder.unstable_update,
    )
    agent_conn = Connection(tap, agent_writer, agent_reader)
    try:
        yield Wire(client, agent_conn)
    finally:
        await agent_conn.close()
        await client.close()


async def settle(recorder: Recorder, count: int) -> None:
    async with asyncio.timeout(5):
        while len(recorder.received) < count:
            await asyncio.sleep(0.01)


# -- The tripwires ---------------------------------------------------------------


def test_acp_library_rejects_subagent_update():
    notification = {
        "sessionId": "root",
        "update": subagent_update("child", state={"state": "running"}),
    }

    with pytest.raises(ValidationError):
        SessionNotification.model_validate(notification)
        pytest.fail(LIBRARY_CAUGHT_UP)


def test_acp_library_has_no_subagents_capability():
    assert "subagents" not in ClientCapabilities.model_fields, LIBRARY_CAUGHT_UP


# -- initialize ------------------------------------------------------------------


async def test_initialize_puts_subagents_capability_on_the_wire():
    agent = Agent()
    async with wired(Recorder(), agent) as wire:
        await wire.client.initialize(
            protocol_version=1, client_capabilities=SUBAGENT_CLIENT_CAPABILITIES
        )

    assert agent.initialize_params["clientCapabilities"] == {
        "auth": {},
        "subagents": {},
    }


async def test_initialize_without_subagent_capabilities_is_the_library_call():
    agent = Agent()
    async with wired(Recorder(), agent) as wire:
        response = await wire.client.initialize(protocol_version=1)

    library_call = InitializeRequest(
        protocol_version=1, client_capabilities=ClientCapabilities()
    )
    assert response.protocol_version == 1
    assert agent.initialize_params == serialize_params(library_call)


# -- Routing ---------------------------------------------------------------------


async def test_unstable_updates_reach_the_callback_in_wire_order():
    sent = [
        ("root", message_chunk("a")),
        ("root", subagent_update("child", title="Child")),
        ("root", session_message("m1", "task")),
        ("child", message_chunk("b")),
        ("root", subagent_update("child", state={"state": "running"})),
        ("child", message_chunk("c")),
        ("child", session_message("m2", "answer")),
        ("root", message_chunk("d")),
        ("root", subagent_update("child", state={"state": "idle"})),
    ]
    recorder = Recorder()
    async with wired(recorder) as wire:
        for session_id, update in sent:
            await wire.send(session_id, update)
        await settle(recorder, len(sent))

    assert [
        (session_id, update.session_update) for session_id, update in recorder.received
    ] == [(session_id, update["sessionUpdate"]) for session_id, update in sent]


async def test_stable_updates_still_reach_the_library_router():
    recorder = Recorder()
    async with wired(recorder) as wire:
        await wire.send("child", message_chunk("hello"))
        await settle(recorder, 1)

    [(session_id, update)] = recorder.received
    assert session_id == "child"
    assert isinstance(update, AgentMessageChunk)
    assert update.content.text == "hello"


async def test_malformed_unstable_update_is_dropped_with_a_warning(caplog):
    malformed = {"sessionUpdate": "subagent_update", "title": "no session id"}
    recorder = Recorder()
    with caplog.at_level(logging.WARNING):
        async with wired(recorder) as wire:
            await wire.send("root", malformed)
            await wire.send("root", message_chunk("after"))
            await settle(recorder, 1)

    assert [type(update) for _, update in recorder.received] == [AgentMessageChunk]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "subagent_update" in warnings[0].getMessage()


def test_patch_fields_tell_omitted_from_null():
    cleared = SubagentUpdate.model_validate(subagent_update("child", title=None))
    omitted = SubagentUpdate.model_validate(subagent_update("child"))

    assert "title" in cleared.model_fields_set
    assert "title" not in omitted.model_fields_set


def test_custom_state_is_kept_whole():
    update = SubagentUpdate.model_validate(
        subagent_update("child", state={"state": "_reviewing", "x": 1})
    )

    assert update.state is not None
    assert update.state.state == "_reviewing"
    assert update.state.model_dump()["x"] == 1


def test_message_content_keeps_non_text_blocks_typed():
    message = SessionMessage.model_validate(
        {
            **session_message("m1", "see"),
            "content": [
                {"type": "text", "text": "see"},
                {"type": "image", "data": "AAAA", "mimeType": "image/png"},
            ],
        }
    )

    assert message.content is not None
    assert [block.type for block in message.content] == ["text", "image"]
