"""Normalizing ACP commands and config options into the SDK's stable DTOs."""

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


def test_a_command_without_input_has_none():
    raw = AvailableCommand(name="summarize", description="Summarize the input")

    assert ACPAvailableCommand.from_protocol(raw) == ACPAvailableCommand(
        name="summarize", description="Summarize the input"
    )


def test_nameless_commands_are_dropped_and_the_rest_kept_in_order():
    raw = [
        AvailableCommand(name="first", description="1"),
        AvailableCommand(name="", description="empty"),
        AvailableCommand(name="second", description="2"),
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

    assert ACPConfigOption.from_protocol(raw).options == [
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


def test_a_category_that_is_not_an_acp_string_category_becomes_none():
    raw = _select(current_value="a", category={"custom": "x"}, options=[])

    assert ACPConfigOption.from_protocol(raw).category is None
