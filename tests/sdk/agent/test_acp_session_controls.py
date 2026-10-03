"""ACP session controls: recording, publishing and setting config options.

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
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from acp.schema import (
    AvailableCommand,
    AvailableCommandsUpdate,
    ConfigOptionUpdate,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
)
from pydantic import ValidationError

from openhands.sdk.agent.acp_agent import (
    ACPAgent,
    ACPConfigOptionRejectedError,
    _apply_config_options,
    _OpenHandsACPBridge,
)
from openhands.sdk.agent.acp_models import (
    ACPAvailableCommand,
    ACPCommandInput,
    ACPConfigOption,
    ACPConfigOptionValue,
    ACPSessionControls,
)
from openhands.sdk.conversation import LocalConversation
from openhands.sdk.conversation.secret_registry import SecretRegistry
from openhands.sdk.conversation.state import (
    ConversationExecutionStatus,
    ConversationState,
)
from openhands.sdk.event import ACPSessionControlsEvent, Event
from openhands.sdk.event.conversation_error import ConversationErrorEvent
from openhands.sdk.utils.async_executor import AsyncExecutor
from openhands.sdk.workspace import LocalWorkspace
from tests.conftest import scripted_acp_command


SUMMARIZE = ACPAvailableCommand(name="summarize", description="Summarize the input")
COMPARE = ACPAvailableCommand(
    name="compare",
    description="Compare two things",
    input=ACPCommandInput(hint="what to compare"),
)


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
THOROUGH = ACPSessionControls(
    available_commands=[SUMMARIZE, COMPARE],
    config_options=[profile_option("thorough", "fast", "thorough")],
)


def command(name: str, description: str = "") -> AvailableCommand:
    return AvailableCommand(name=name, description=description)


def wait_until(condition: Callable[[], Any], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.02)


def controls_events(events: Any) -> list[ACPSessionControlsEvent]:
    return [e for e in events if isinstance(e, ACPSessionControlsEvent)]


def methods(log: list[dict[str, Any]]) -> list[str]:
    return [entry["method"] for entry in log]


class Started:
    """An ACP agent started on a state of its own, publishing into a list."""

    def __init__(self, agent: ACPAgent, state: ConversationState) -> None:
        self.agent = agent
        self.state = state
        self.published: list[Event] = []
        self.emitted: list[Event] = []


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
        run = Started(agent, state)
        agent._on_session_event = run.published.append
        started.append(run)
        agent.init_state(state, on_event=run.emitted.append)
        return run

    yield _start
    for run in started:
        run.agent.close()


@pytest.fixture
def conversation(tmp_path: Path) -> Iterator[Callable[..., LocalConversation]]:
    conversations: list[LocalConversation] = []
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)

    def _conversation(*flags: str, **fields: Any) -> LocalConversation:
        agent = ACPAgent(acp_command=scripted_acp_command(*flags), **fields)
        conv = LocalConversation(
            agent,
            workspace=str(workspace),
            persistence_dir=str(tmp_path / "conversations"),
            visualizer=None,
        )
        conversations.append(conv)
        return conv

    yield _conversation
    for conv in conversations:
        conv.close()


def bridged_agent(session_id: str = "root") -> tuple[ACPAgent, _OpenHandsACPBridge]:
    """An agent wired to a bridge with no process behind it."""
    agent = ACPAgent(acp_command=["unused"])
    bridge = _OpenHandsACPBridge()
    agent._client = bridge
    agent._session_id = session_id
    agent._bind_session_controls()
    return agent, bridge


# -- Recording and publishing -------------------------------------------------


def test_controls_reported_while_the_session_starts_are_published_once_it_started(
    start,
):
    run = start(acp_config_options={"profile": "thorough"})

    assert [e.controls for e in controls_events(run.published)] == [THOROUGH]
    assert run.agent.session_controls == THOROUGH


def test_commands_reported_after_session_new_answered_are_published(start):
    run = start()

    wait_until(lambda: run.agent.session_controls.available_commands)

    wait_until(lambda: controls_events(run.published)[-1].controls == FAST)


def test_each_session_keeps_its_own_controls_and_only_the_root_is_published():
    agent, bridge = bridged_agent("root")
    published: list[Event] = []
    agent._on_session_event = published.append

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
    assert bridge.wait_for_available_commands("root", timeout=0)


def test_agent_supplied_text_is_masked_before_it_is_stored():
    registry = SecretRegistry()
    registry.update_secrets({"API_TOKEN": "tok-12345"})
    agent, bridge = bridged_agent()
    bridge.mask = registry.mask_secrets_in_output
    published: list[Event] = []
    agent._on_session_event = published.append

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
    agent, bridge = bridged_agent()
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
    agent, bridge = bridged_agent()
    published: list[Event] = []
    agent._on_session_event = published.append
    agent._starting_session = True

    bridge.record_available_commands("root", [command("early")])

    assert published == []


# -- Option values at the start ------------------------------------------------


def test_start_values_reach_the_agent_after_session_new_and_before_the_prompt(
    conversation, acp_request_log
):
    conv = conversation(acp_config_options={"profile": "thorough"})
    conv.send_message("hello")

    conv.run()

    calls = methods(acp_request_log())
    assert calls.index("session/new") < calls.index("session/set_config_option")
    assert calls.index("session/set_config_option") < calls.index("session/prompt")
    set_params = acp_request_log()[calls.index("session/set_config_option")]["params"]
    assert (set_params["configId"], set_params["value"]) == ("profile", "thorough")
    wait_until(
        lambda: (
            controls_events(conv.state.events)[-1].config_options[0].current_value
            == "thorough"
        )
    )


async def test_values_are_set_in_order_and_every_response_is_recorded():
    responses = {
        "a": SimpleNamespace(config_options=["after-a"]),
        "b": SimpleNamespace(config_options=["after-b"]),
    }
    conn = AsyncMock()
    conn.set_config_option.side_effect = lambda config_id, **_: responses[config_id]
    recorded: list[tuple[str, list[str]]] = []

    await _apply_config_options(
        conn,
        "s1",
        {"b": "2", "a": True},
        on_config_options=lambda sid, options: recorded.append((sid, list(options))),
        mask=lambda text: text,
    )

    assert [c.kwargs["config_id"] for c in conn.set_config_option.await_args_list] == [
        "b",
        "a",
    ]
    assert recorded == [("s1", ["after-b"]), ("s1", ["after-a"])]


def test_a_refused_start_value_ends_the_start_and_no_prompt_is_sent(
    conversation, acp_request_log
):
    conv = conversation(acp_config_options={"profile": "turbo"})
    conv.send_message("hello")

    with pytest.raises(ACPConfigOptionRejectedError):
        conv.run()

    errors = [e for e in conv.state.events if isinstance(e, ConversationErrorEvent)]
    assert [(e.code, e.detail) for e in errors] == [
        ("ACPConfigOptionRejected", "unknown profile 'turbo'")
    ]
    assert conv.state.execution_status == ConversationExecutionStatus.ERROR
    assert "session/prompt" not in methods(acp_request_log())


def test_after_a_successful_load_no_value_is_reapplied(
    start, acp_request_log, tmp_path
):
    sessions = str(tmp_path / "sessions.json")
    conversation_id = uuid.uuid4()
    persistence_dir = tmp_path / "persisted"
    first = start(
        "--sessions-file",
        sessions,
        conversation_id=conversation_id,
        persistence_dir=persistence_dir,
        acp_config_options={"profile": "thorough"},
    )
    first.agent.close()
    before_resume = len(acp_request_log())

    resumed = start(
        "--sessions-file",
        sessions,
        conversation_id=conversation_id,
        persistence_dir=persistence_dir,
        acp_config_options={"profile": "thorough"},
    )

    resumed_calls = methods(acp_request_log()[before_resume:])
    assert "session/load" in resumed_calls
    assert "session/new" not in resumed_calls
    assert "session/set_config_option" not in resumed_calls
    assert resumed.agent._session_id == first.agent._session_id
    wait_until(lambda: resumed.agent.session_controls == THOROUGH)


def test_after_a_fallback_to_a_fresh_session_every_value_is_reapplied(
    start, acp_request_log, tmp_path
):
    conversation_id = uuid.uuid4()
    persistence_dir = tmp_path / "persisted"
    first = start(
        conversation_id=conversation_id,
        persistence_dir=persistence_dir,
        acp_config_options={"profile": "thorough"},
    )
    first.agent.close()
    before_resume = len(acp_request_log())

    # A new process does not know the old session, so session/load fails.
    resumed = start(
        conversation_id=conversation_id,
        persistence_dir=persistence_dir,
        acp_config_options={"profile": "thorough"},
    )

    resumed_calls = methods(acp_request_log()[before_resume:])
    assert resumed_calls.index("session/load") < resumed_calls.index("session/new")
    assert "session/set_config_option" in resumed_calls
    assert resumed.agent.session_controls == THOROUGH


# -- The model option ------------------------------------------------------------


@pytest.mark.parametrize("config_id", ["model", ""])
def test_the_model_option_and_an_empty_id_are_refused_in_the_field(config_id):
    with pytest.raises(ValidationError):
        ACPAgent(acp_command=["unused"], acp_config_options={config_id: "x"})


def test_a_model_switch_through_set_config_option_updates_the_published_model():
    agent, bridge = bridged_agent()
    published: list[Event] = []
    agent._on_session_event = published.append
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
    conn.set_config_option.return_value = SimpleNamespace(config_options=[model_option])
    agent._conn = conn
    agent._executor = AsyncExecutor()
    agent._model_via_config_option = True
    try:
        agent.set_acp_model("m2")
    finally:
        agent._executor.close()

    model = controls_events(published)[-1].config_options[0]
    assert (model.id, model.current_value) == ("model", "m2")


# -- Setting an option live -----------------------------------------------------


def test_a_live_set_returns_the_agents_new_controls(start):
    run = start()

    controls = run.agent.set_acp_config_option("profile", "thorough")

    assert controls == THOROUGH
    wait_until(lambda: controls_events(run.published)[-1].controls == THOROUGH)


def test_a_set_before_any_session_is_refused():
    agent = ACPAgent(acp_command=scripted_acp_command())

    with pytest.raises(RuntimeError):
        agent.set_acp_config_option("profile", "thorough")
