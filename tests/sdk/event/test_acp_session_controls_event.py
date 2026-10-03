"""ACPSessionControlsEvent: persisted state of an ACP session's controls."""

import json

from openhands.sdk.agent.acp_models import (
    ACPAvailableCommand,
    ACPCommandInput,
    ACPConfigOption,
    ACPConfigOptionValue,
    ACPSessionControls,
)
from openhands.sdk.event import (
    ACPSessionControlsEvent,
    Event,
)


CONTROLS = ACPSessionControls(
    available_commands=[
        ACPAvailableCommand(name="summarize", description="Summarize the input"),
        ACPAvailableCommand(
            name="compare",
            description="Compare two things",
            input=ACPCommandInput(hint="what to compare"),
        ),
    ],
    config_options=[
        ACPConfigOption(
            id="profile",
            name="Profile",
            type="select",
            current_value="thorough",
            options=[
                ACPConfigOptionValue(value="fast", name="fast"),
                ACPConfigOptionValue(value="thorough", name="thorough"),
            ],
        ),
        ACPConfigOption(
            id="verbose", name="Verbose", type="boolean", current_value=False
        ),
    ],
)


def test_event_round_trips_through_json_as_its_own_kind():
    event = ACPSessionControlsEvent.from_controls(CONTROLS)

    payload = json.loads(event.model_dump_json())
    restored = Event.model_validate_json(event.model_dump_json())

    assert payload["kind"] == "ACPSessionControlsEvent"
    assert payload["source"] == "agent"
    assert isinstance(restored, ACPSessionControlsEvent)
    assert restored.controls == CONTROLS


def test_event_renders_as_one_line_of_command_names_and_option_values():
    event = ACPSessionControlsEvent.from_controls(CONTROLS)

    assert event.visualize.plain == (
        "Commands: /summarize /compare | Options: profile=thorough, verbose=False"
    )
    assert "\n" not in str(event)


def test_an_empty_event_still_renders_one_line():
    assert ACPSessionControlsEvent().visualize.plain == (
        "Commands: none | Options: none"
    )
