"""LocalConversation.set_acp_config_option and its out-of-turn event emitter."""

from __future__ import annotations

import gc
import threading
import time
import weakref
from typing import Any

import pytest
from acp.schema import AvailableCommand

from openhands.sdk.agent.acp_agent import ACPAgent, ACPConfigOptionRejectedError
from openhands.sdk.event import ACPSessionControlsEvent, ActionEvent, PauseEvent
from tests.conftest import controls_events, wait_until


def test_a_set_before_the_start_is_persisted_and_applied_at_the_start(
    scripted_conversation, acp_request_log
):
    conv = scripted_conversation()

    assert conv.set_acp_config_option("profile", "thorough") is None
    assert isinstance(conv.agent, ACPAgent)
    assert conv.agent.acp_config_options == {"profile": "thorough"}
    assert conv.state.agent.acp_config_options == {"profile": "thorough"}
    assert acp_request_log() == []

    conv.send_message("hello")
    conv.run()

    sets = [e for e in acp_request_log() if e["method"] == "session/set_config_option"]
    assert [(e["params"]["configId"], e["params"]["value"]) for e in sets] == [
        ("profile", "thorough")
    ]
    wait_until(
        lambda: controls_events(conv.state.events)[-1].config_options[0].current_value
        == "thorough"
    )


def test_a_live_set_is_persisted_and_survives_a_reload(scripted_conversation):
    conv = scripted_conversation()
    conv.send_message("hello")
    conv.run()

    with pytest.raises(ACPConfigOptionRejectedError) as fixed:
        conv.set_acp_config_option("profile", "thorough")
    controls = conv.set_acp_config_option("profile", "fast")

    assert (
        str(fixed.value)
        == "profile is fixed once the session has started (it is 'fast')"
    )
    assert (fixed.value.config_id, fixed.value.value) == ("profile", "thorough")
    assert controls is not None
    assert controls.config_options[0].current_value == "fast"
    conversation_id = conv.id
    conv.close()
    reloaded = scripted_conversation(conversation_id=conversation_id)
    assert isinstance(reloaded.agent, ACPAgent)
    assert reloaded.agent.acp_config_options == {"profile": "fast"}


def test_a_refused_live_set_writes_nothing(scripted_conversation):
    conv = scripted_conversation()
    conv.send_message("hello")
    conv.run()
    agent_before = conv.agent

    with pytest.raises(ACPConfigOptionRejectedError):
        conv.set_acp_config_option("profile", "turbo")

    assert conv.agent is agent_before
    assert isinstance(conv.agent, ACPAgent)
    assert conv.agent.acp_config_options == {}


def test_the_agent_swap_hands_publishing_to_the_copy(scripted_conversation):
    conv = scripted_conversation()
    conv.send_message("hello")
    conv.run()
    old_agent = weakref.ref(conv.agent)

    conv.set_acp_config_option("profile", "fast")
    gc.collect()

    # The shared bridge does not keep the replaced agent alive ...
    assert old_agent() is None
    agent = conv.agent
    assert isinstance(agent, ACPAgent)
    assert agent.has_live_acp_session
    # ... and what it records next is published through the copy.
    assert agent._session_id is not None
    agent._client.record_available_commands(
        agent._session_id, [AvailableCommand(name="late", description="")]
    )
    wait_until(
        lambda: [
            c.name for c in controls_events(conv.state.events)[-1].available_commands
        ]
        == ["late"]
    )


def test_a_portal_thread_event_during_a_synchronous_run_lands_after_the_step(
    scripted_conversation,
):
    conv = scripted_conversation()
    conv.send_message("hello")
    finished = threading.Event()

    def run() -> None:
        conv.run()
        finished.set()

    threading.Thread(target=run, daemon=True).start()

    assert finished.wait(30), "run() deadlocked"

    def narrowed(event: Any) -> bool:
        # Only the first prompt narrows the profile to its current value.
        return (
            isinstance(event, ACPSessionControlsEvent)
            and len(event.config_options[0].options) == 1
        )

    wait_until(lambda: any(narrowed(e) for e in conv.state.events))
    events = list(conv.state.events)
    narrowed_at = next(i for i, e in enumerate(events) if narrowed(e))
    step_finished_at = max(
        i for i, e in enumerate(events) if isinstance(e, ActionEvent)
    )
    assert narrowed_at > step_finished_at


def test_events_emitted_after_close_are_dropped(scripted_conversation):
    conv = scripted_conversation()
    conv.close()

    conv._emit_event_from_any_thread(PauseEvent())

    time.sleep(0.1)
    assert not any(isinstance(e, PauseEvent) for e in conv.state.events)


def test_events_from_other_threads_are_persisted_in_submission_order(
    scripted_conversation,
):
    conv = scripted_conversation()
    sent = [PauseEvent() for _ in range(50)]

    for event in sent:
        conv._emit_event_from_any_thread(event)

    wait_until(lambda: sum(isinstance(e, PauseEvent) for e in conv.state.events) == 50)
    persisted = [e.id for e in conv.state.events if isinstance(e, PauseEvent)]
    assert persisted == [e.id for e in sent]
