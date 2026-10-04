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
    InitializeResponse,
    SessionNotification,
)
from pydantic import ValidationError

from openhands.sdk.agent.acp_unstable import (
    SessionMessage,
    SubagentClientSideConnection,
    UnstableSessionUpdate,
)
from tests.fixtures.acp.scripted_agent import message, said, subagent, text


LIBRARY_CAUGHT_UP = (
    "agent-client-protocol now parses ACP's sub-agent updates: delete "
    "openhands/sdk/agent/acp_unstable.py, use the library's types and capability, "
    "re-run the ACP conformance probes against Claude Code, Codex and Gemini, and "
    "bump agent-client-protocol in openhands-sdk/pyproject.toml."
)


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
        "update": subagent("child", state={"state": "running"}),
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
        await wire.client.initialize(protocol_version=1)

    assert agent.initialize_params == {
        "protocolVersion": 1,
        "clientCapabilities": {"auth": {}, "subagents": {}},
    }


# -- Routing ---------------------------------------------------------------------


async def test_unstable_updates_reach_the_callback_in_wire_order():
    sent = [
        ("root", said("a")),
        ("root", subagent("child", title="Child")),
        ("root", message("m1", "root", "child", "task")),
        ("child", said("b")),
        ("root", subagent("child", state={"state": "running"})),
        ("child", said("c")),
        ("child", message("m2", "root", "child", "answer")),
        ("root", said("d")),
        ("root", subagent("child", state={"state": "idle"})),
    ]
    recorder = Recorder()
    async with wired(recorder) as wire:
        for session_id, update in sent:
            await wire.send(session_id, update)
        await settle(recorder, len(sent))

    assert [
        (session_id, update.session_update) for session_id, update in recorder.received
    ] == [(session_id, update["sessionUpdate"]) for session_id, update in sent]


async def test_session_message_with_a_non_text_block_reaches_the_callback_whole():
    image = {"type": "image", "data": "AAAA", "mimeType": "image/png"}
    recorder = Recorder()
    async with wired(recorder) as wire:
        await wire.send(
            "root", message("m1", "root", "child", "see", content=[text("see"), image])
        )
        await settle(recorder, 1)

    [(_, update)] = recorder.received
    assert isinstance(update, SessionMessage)
    assert [block.type for block in update.content or []] == ["text", "image"]


async def test_stable_updates_still_reach_the_library_router():
    recorder = Recorder()
    async with wired(recorder) as wire:
        await wire.send("child", said("hello"))
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
            await wire.send("root", said("after"))
            await settle(recorder, 1)

    assert [type(update) for _, update in recorder.received] == [AgentMessageChunk]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "subagent_update" in warnings[0].getMessage()
