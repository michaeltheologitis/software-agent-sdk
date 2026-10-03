"""Normalizing ACP commands and config options into the SDK's stable DTOs."""

from acp.schema import (
    AvailableCommand,
    AvailableCommandInput,
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


def test_a_category_that_is_not_an_acp_string_category_becomes_none():
    raw = _select(current_value="a", category={"custom": "x"}, options=[])

    assert ACPConfigOption.from_protocol(raw).category is None
