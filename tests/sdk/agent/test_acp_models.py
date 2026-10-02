"""Normalizing ACP commands and config options into the SDK's stable DTOs."""

from types import SimpleNamespace

import pytest
from acp.schema import (
    AvailableCommand,
    AvailableCommandInput,
    SessionConfigOptionBoolean,
    SessionConfigOptionSelect,
    SessionConfigSelectGroup,
    SessionConfigSelectOption,
    UnstructuredCommandInput,
)

from openhands.sdk.agent.acp_models import (
    ACPAvailableCommand,
    ACPCommandInput,
    ACPConfigOption,
    ACPConfigOptionValue,
    ACPSessionControls,
)


def _select(**fields) -> SessionConfigOptionSelect:
    return SessionConfigOptionSelect(
        **{"type": "select", "id": "profile", "name": "Profile", **fields}
    )


def test_command_hint_is_read_through_the_root_model():
    raw = AvailableCommand(
        name="compare",
        description="Compare two things",
        input=AvailableCommandInput(UnstructuredCommandInput(hint="what to compare")),
    )

    assert ACPAvailableCommand.from_protocol(raw) == ACPAvailableCommand(
        name="compare",
        description="Compare two things",
        input=ACPCommandInput(hint="what to compare"),
    )


@pytest.mark.parametrize(
    "raw, expected",
    [
        (
            SimpleNamespace(name="summarize", input=None),
            ACPAvailableCommand(name="summarize", description=""),
        ),
        (
            SimpleNamespace(name="ask", description="Ask", input={"hint": 3}),
            ACPAvailableCommand(name="ask", description="Ask"),
        ),
        (
            SimpleNamespace(name="ask", description="Ask", input=SimpleNamespace()),
            ACPAvailableCommand(name="ask", description="Ask"),
        ),
    ],
)
def test_command_degrades_to_an_empty_description_and_no_input(raw, expected):
    assert ACPAvailableCommand.from_protocol(raw) == expected


def test_nameless_commands_are_dropped_and_the_rest_kept_in_order():
    raw = [
        SimpleNamespace(name="first", description="1"),
        SimpleNamespace(name="", description="empty"),
        SimpleNamespace(description="missing"),
        SimpleNamespace(name=7, description="not a string"),
        SimpleNamespace(name="second", description="2"),
    ]

    assert [c.name for c in ACPSessionControls.parse_commands(raw)] == [
        "first",
        "second",
    ]


def test_grouped_select_is_flattened_with_each_value_keeping_its_group():
    raw = _select(
        current_value="fast",
        options=[
            SessionConfigSelectGroup(
                group="g1",
                name="Speed",
                options=[SessionConfigSelectOption(value="fast", name="Fast")],
            ),
            SessionConfigSelectGroup(
                group="g2",
                name="Depth",
                options=[
                    SessionConfigSelectOption(
                        value="thorough", name="Thorough", description="Slow"
                    )
                ],
            ),
        ],
    )

    option = ACPConfigOption.from_protocol(raw)

    assert option is not None
    assert option.options == [
        ACPConfigOptionValue(value="fast", name="Fast", group="Speed"),
        ACPConfigOptionValue(
            value="thorough", name="Thorough", description="Slow", group="Depth"
        ),
    ]


def test_ungrouped_select_keeps_values_in_order_without_a_group():
    raw = _select(
        current_value="b",
        category="mode",
        description="Pick one",
        options=[
            SessionConfigSelectOption(value="a", name="A"),
            SessionConfigSelectOption(value="b", name="B"),
        ],
    )

    assert ACPConfigOption.from_protocol(raw) == ACPConfigOption(
        id="profile",
        name="Profile",
        type="select",
        current_value="b",
        description="Pick one",
        category="mode",
        options=[
            ACPConfigOptionValue(value="a", name="A"),
            ACPConfigOptionValue(value="b", name="B"),
        ],
    )


def test_boolean_option_keeps_its_boolean_value_and_has_no_values():
    raw = SessionConfigOptionBoolean(
        type="boolean", id="verbose", name="Verbose", current_value=True
    )

    assert ACPConfigOption.from_protocol(raw) == ACPConfigOption(
        id="verbose", name="Verbose", type="boolean", current_value=True
    )


def test_option_wrapped_in_a_root_model_is_unwrapped():
    inner = _select(
        current_value="a", options=[SessionConfigSelectOption(value="a", name="A")]
    )

    option = ACPConfigOption.from_protocol(SimpleNamespace(root=inner))

    assert option is not None
    assert option.current_value == "a"


def test_a_category_that_is_not_an_acp_string_category_becomes_none():
    raw = _select(current_value="a", category={"custom": "x"}, options=[])

    option = ACPConfigOption.from_protocol(raw)

    assert option is not None
    assert option.category is None


@pytest.mark.parametrize(
    "raw",
    [
        SimpleNamespace(type="text", id="note", name="Note", current_value="x"),
        SimpleNamespace(type="select", id="", name="Empty id", current_value="x"),
        SimpleNamespace(type="select", id="s", name="No value", current_value=None),
        SimpleNamespace(type="boolean", id="b", name="Not bool", current_value="yes"),
    ],
)
def test_unusable_options_are_dropped_not_raised(raw):
    good = _select(current_value="a", options=[])

    parsed = ACPSessionControls.parse_config_options([raw, good])

    assert [o.id for o in parsed] == ["profile"]
