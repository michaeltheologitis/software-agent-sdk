"""LocalConversation's out-of-turn event emitter, which ACP session controls use."""

from __future__ import annotations

import threading
import time
from typing import Any

from openhands.sdk.event import ACPSessionControlsEvent, ActionEvent, PauseEvent
from tests.conftest import wait_until


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
