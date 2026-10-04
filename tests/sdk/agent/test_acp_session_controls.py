"""ACP session controls: recording and publishing.

The outside world here is the ACP agent process, so most tests run a real
one: the scripted test agent in ``tests/fixtures/acp/scripted_agent.py``. It
offers a ``profile`` option (``fast`` or ``thorough``) and commands that depend
on it; its first prompt clears the commands and fixes the profile.
"""

from __future__ import annotations

import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import AsyncMock

import pytest
from acp.schema import (
    AvailableCommand,
    AvailableCommandsUpdate,
    ConfigOptionUpdate,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SetSessionConfigOptionResponse,
)

from openhands.sdk.agent.acp_agent import ACPAgent, _OpenHandsACPBridge
from openhands.sdk.agent.acp_models import (
    ACPAvailableCommand,
    ACPConfigOption,
    ACPConfigOptionValue,
    ACPSessionControls,
)
from openhands.sdk.conversation.secret_registry import SecretRegistry
from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.event import ACPSessionControlsEvent, Event
from openhands.sdk.utils.async_executor import AsyncExecutor
from openhands.sdk.workspace import LocalWorkspace
from tests.conftest import controls_events, scripted_acp_command, wait_until


SUMMARIZE = ACPAvailableCommand(name="summarize", description="Summarize the input")


def profile_option(current: str, *values: str) -> ACPConfigOption:
    return ACPConfigOption(
        id="profile",
        name="Profile",
        type="select",
        current_value=current,
        options=[ACPConfigOptionValue(value=v, name=v) for v in values],
    )


FAST = ACPSessionControls(
    available_commands=[SUMMARIZE],
    config_options=[profile_option("fast", "fast", "thorough")],
)


def command(name: str, description: str = "") -> AvailableCommand:
    return AvailableCommand(name=name, description=description)


class Started(NamedTuple):
    """An ACP agent started on a state of its own, publishing into a list."""

    agent: ACPAgent
    published: list[Event]


@pytest.fixture
def start(tmp_path: Path) -> Iterator[Callable[..., Started]]:
    """Start a scripted agent through ACPAgent.init_state, without a conversation."""
    started: list[Started] = []
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def _start(
        *flags: str,
        conversation_id: uuid.UUID | None = None,
        persistence_dir: Path | None = None,
        **fields: Any,
    ) -> Started:
        agent = ACPAgent(acp_command=scripted_acp_command(*flags), **fields)
        state = ConversationState.create(
            id=conversation_id or uuid.uuid4(),
            agent=agent,
            workspace=LocalWorkspace(working_dir=str(workspace)),
            persistence_dir=str(persistence_dir or tmp_path / uuid.uuid4().hex),
        )
        run = Started(agent, [])
        agent._on_session_event = run.published.append
        started.append(run)
        agent.init_state(state, on_event=lambda _event: None)
        return run

    yield _start
    for run in started:
        run.agent.close()


def bridged_agent() -> tuple[ACPAgent, _OpenHandsACPBridge, list[Event]]:
    """An agent on session "root" of a bridge with no process, and what it publishes."""
    agent = ACPAgent(acp_command=["unused"])
    bridge = _OpenHandsACPBridge()
    agent._client = bridge
    agent._session_id = "root"
    agent._bind_session_controls()
    published: list[Event] = []
    agent._on_session_event = published.append
    return agent, bridge, published


# -- Recording and publishing -------------------------------------------------


def test_commands_reported_after_session_new_answered_are_published(start):
    run = start()

    wait_until(lambda: controls_events(run.published)[-1].controls == FAST)


def test_each_session_keeps_its_own_controls_and_only_the_root_is_published():
    _, bridge, published = bridged_agent()

    bridge.record_available_commands("root", [command("root-cmd")])
    bridge.record_available_commands("child", [command("child-cmd")])

    assert [c.name for c in bridge.session_controls("child").available_commands] == [
        "child-cmd"
    ]
    assert [
        [c.name for c in e.available_commands] for e in controls_events(published)
    ] == [["root-cmd"]]


async def test_entries_the_protocol_cannot_parse_are_dropped_not_raised():
    bridge = _OpenHandsACPBridge()
    commands = AvailableCommandsUpdate.model_validate(
        {
            "sessionUpdate": "available_commands_update",
            "availableCommands": [
                {"name": "no-description"},
                {"name": "", "description": "no name"},
                {"name": "ask", "description": "Ask", "input": {"hint": 3}},
                {"name": "go", "description": "Go"},
            ],
        }
    )
    options = ConfigOptionUpdate.model_validate(
        {
            "sessionUpdate": "config_option_update",
            "configOptions": [
                {"type": "text", "id": "note", "name": "Note", "currentValue": "x"},
                {"type": "boolean", "id": "verbose", "name": "V", "currentValue": True},
            ],
        }
    )

    await bridge.session_update("root", commands)
    await bridge.session_update("root", options)

    assert bridge.session_controls("root") == ACPSessionControls(
        available_commands=[
            ACPAvailableCommand(name="ask", description="Ask"),
            ACPAvailableCommand(name="go", description="Go"),
        ],
        config_options=[
            ACPConfigOption(id="verbose", name="V", type="boolean", current_value=True)
        ],
    )


async def test_session_updates_of_both_kinds_are_recorded_and_not_routed_on():
    bridge = _OpenHandsACPBridge()
    bridge.on_event = AsyncMock()

    await bridge.session_update(
        "root",
        AvailableCommandsUpdate(
            session_update="available_commands_update",
            available_commands=[AvailableCommand(name="go", description="Go")],
        ),
    )

    assert bridge.session_controls("root").available_commands == [
        ACPAvailableCommand(name="go", description="Go")
    ]
    bridge.on_event.assert_not_called()


def test_agent_supplied_text_is_masked_before_it_is_stored():
    registry = SecretRegistry()
    registry.update_secrets({"API_TOKEN": "tok-12345"})
    _, bridge, published = bridged_agent()
    bridge.mask = registry.mask_secrets_in_output

    bridge.record_available_commands("root", [command("leak", "uses tok-12345")])
    bridge.record_config_options(
        "root",
        [
            SessionConfigOptionSelect(
                type="select",
                id="profile",
                name="Profile",
                description="key tok-12345",
                current_value="a",
                options=[SessionConfigSelectOption(value="a", name="tok-12345")],
            )
        ],
    )

    dumped = controls_events(published)[-1].model_dump_json()
    assert "tok-12345" not in dumped
    assert "<secret-hidden>" in dumped


def test_concurrent_publishes_keep_snapshot_order_and_end_on_the_newest():
    agent, bridge, _ = bridged_agent()
    published: list[ACPSessionControlsEvent] = []

    def slow_sink(event: Event) -> None:
        time.sleep(0.0002)
        assert isinstance(event, ACPSessionControlsEvent)
        published.append(event)

    agent._on_session_event = slow_sink
    bridge.record_available_commands("root", [command("0")])

    def record() -> None:
        for i in range(1, 300):
            bridge.record_available_commands("root", [command(f"{i}")])

    def publish() -> None:
        for _ in range(300):
            agent._publish_session_controls()

    threads = [threading.Thread(target=f) for f in (record, publish, publish)]
    # Switch threads as often as possible, so unordered publishing would show.
    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        sys.setswitchinterval(switch_interval)

    sequence = [int(e.available_commands[0].name) for e in published]
    assert sequence == sorted(set(sequence))
    assert published[-1].controls == bridge.session_controls("root")


def test_nothing_is_published_while_a_session_is_starting():
    agent, bridge, published = bridged_agent()
    agent._starting_session = True

    bridge.record_available_commands("root", [command("early")])

    assert published == []


# -- The model option ------------------------------------------------------------


def test_a_model_switch_through_set_config_option_updates_the_published_model():
    agent, bridge, published = bridged_agent()
    model_option = SessionConfigOptionSelect(
        type="select",
        id="model",
        name="Model",
        current_value="m2",
        options=[
            SessionConfigSelectOption(value="m1", name="M1"),
            SessionConfigSelectOption(value="m2", name="M2"),
        ],
    )
    conn = AsyncMock()
    conn.set_config_option.return_value = SetSessionConfigOptionResponse(
        config_options=[model_option]
    )
    agent._conn = conn
    agent._executor = AsyncExecutor()
    agent._model_via_config_option = True
    try:
        agent.set_acp_model("m2")
    finally:
        agent._executor.close()

    model = controls_events(published)[-1].config_options[0]
    assert (model.id, model.current_value) == ("model", "m2")
