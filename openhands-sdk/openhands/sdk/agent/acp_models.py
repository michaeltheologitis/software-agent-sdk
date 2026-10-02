"""Stable DTOs for ACP session metadata: models, commands and config options.

These live in a standalone module — *not* ``acp_agent`` — so the agent-server
can import them for its public ``ConversationInfo`` schema without importing
``ACPAgent``, which would eagerly register it in the agent
``DiscriminatedUnion`` (see ``openhands/sdk/agent/__init__.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


class ACPModelInfo(BaseModel):
    """One model an ACP server offers for a session.

    A normalized, stable mirror of the ACP protocol's ``ModelInfo``. The
    protocol ``models`` capability is flagged **UNSTABLE**, so we re-map it
    into our own type at the SDK boundary rather than re-serializing the
    vendored ``acp.schema`` type onto the agent-server's public API — clients
    get a stable shape regardless of upstream protocol churn.

    Carries everything a client needs to render a picker and resolve a
    ``current_model_id`` to a display label *itself*; the SDK deliberately
    does no name curation.
    """

    # ``model_id`` collides with pydantic's protected ``model_`` namespace;
    # opt out (the name mirrors the protocol field and the persisted shape).
    model_config = ConfigDict(protected_namespaces=())

    model_id: str = Field(
        description=(
            "Server-assigned model identifier. May be concrete "
            '(e.g. ``"gpt-5.6"``) or an opaque alias '
            '(e.g. ``"default"``, ``"auto"``). This is the value to pass back '
            "to the server to switch to this model."
        ),
    )
    name: str | None = Field(
        default=None,
        description='Human-readable label, e.g. ``"GPT-5.5"``.',
    )
    description: str | None = Field(
        default=None,
        description="Optional longer description supplied by the server.",
    )

    @classmethod
    def from_protocol(cls, raw: Any, *, id_attr: str = "model_id") -> ACPModelInfo:
        """Build from a raw ACP ``ModelInfo`` (or any duck-typed object).

        Tolerant of partial/malformed entries: non-string fields degrade to
        ``""`` (``model_id``) or ``None`` (``name``/``description``) rather
        than raising, since the source is an UNSTABLE protocol capability that
        older or half-implemented agents may emit incompletely.

        ``id_attr`` names the attribute carrying the model id — ``"model_id"``
        for a ``models``-capability ``ModelInfo``, ``"value"`` for a
        ``configOptions`` select option.
        """
        model_id = getattr(raw, id_attr, None)
        name = getattr(raw, "name", None)
        description = getattr(raw, "description", None)
        return cls(
            model_id=model_id if isinstance(model_id, str) else "",
            name=name if isinstance(name, str) else None,
            description=description if isinstance(description, str) else None,
        )


ACPConfigOptionType = Literal["select", "boolean"]


class ACPCommandInput(BaseModel):
    """The text a command takes after its name; ACP's ``UnstructuredCommandInput``."""

    hint: str = Field(
        description="Placeholder a client shows until the user types the input.",
    )


class ACPAvailableCommand(BaseModel):
    """One slash command an ACP session offers; ACP's ``AvailableCommand``.

    A client invokes it by sending a user message whose text starts with
    ``/<name>``; the bridge forwards that text unchanged.
    """

    name: str = Field(
        description="Command name, without the leading slash.",
    )
    description: str = Field(
        description="What the command does, in the agent's words.",
    )
    input: ACPCommandInput | None = Field(
        default=None,
        description="Present when the command takes text after its name.",
    )

    @classmethod
    def from_protocol(cls, raw: Any) -> ACPAvailableCommand | None:
        """Build from an ACP ``AvailableCommand``; ``None`` without a usable name."""
        name = getattr(raw, "name", None)
        if not isinstance(name, str) or not name:
            return None
        description = getattr(raw, "description", None)
        raw_input = getattr(raw, "input", None)
        # agent-client-protocol 0.12.1 wraps the input in a RootModel.
        hint = getattr(getattr(raw_input, "root", raw_input), "hint", None)
        return cls(
            name=name,
            description=description if isinstance(description, str) else "",
            input=ACPCommandInput(hint=hint) if isinstance(hint, str) else None,
        )


class ACPConfigOptionValue(BaseModel):
    """One value of a select option; ACP's ``SessionConfigSelectOption``."""

    value: str = Field(
        description="The value to send back in session/set_config_option.",
    )
    name: str = Field(
        description="Human-readable label for the value.",
    )
    description: str | None = Field(
        default=None,
        description="Optional longer description supplied by the agent.",
    )
    group: str | None = Field(
        default=None,
        description="Label of the ACP option group the value came from, if any.",
    )


class ACPConfigOption(BaseModel):
    """One session config option; ACP's ``SessionConfigOptionSelect`` or ``…Boolean``.

    Select groups are flattened into ``options``, each value keeping its
    group's label in ``group``.
    """

    id: str = Field(
        description="The option's id, the configId of session/set_config_option.",
    )
    name: str = Field(
        description="Human-readable label for the option.",
    )
    type: ACPConfigOptionType = Field(
        description="'select' (one of options) or 'boolean'.",
    )
    current_value: str | bool = Field(
        description="The current value: a str for a select, a bool for a boolean.",
    )
    description: str | None = Field(
        default=None,
        description="Optional description for the client to display.",
    )
    category: str | None = Field(
        default=None,
        description="ACP's UX hint: mode, model, model_config or thought_level.",
    )
    options: list[ACPConfigOptionValue] = Field(
        default_factory=list,
        description="The selectable values of a select; empty for a boolean.",
    )

    @classmethod
    def from_protocol(cls, raw: Any) -> ACPConfigOption | None:
        """Build from an ACP config option; ``None`` for a type this model lacks."""
        # Older ACP Python releases wrap each option in a RootModel.
        option = getattr(raw, "root", raw)
        option_type = getattr(option, "type", None)
        option_id = getattr(option, "id", None)
        current = getattr(option, "current_value", None)
        if not isinstance(option_id, str) or not option_id:
            return None
        if option_type == "select" and isinstance(current, str):
            values = _flatten_select_values(getattr(option, "options", None) or [])
        elif option_type == "boolean" and isinstance(current, bool):
            values = []
        else:
            logger.debug(
                "Dropping ACP config option %r of type %r", option_id, option_type
            )
            return None
        name = getattr(option, "name", None)
        description = getattr(option, "description", None)
        category = getattr(option, "category", None)
        return cls(
            id=option_id,
            name=name if isinstance(name, str) else option_id,
            type=option_type,
            current_value=current,
            description=description if isinstance(description, str) else None,
            category=category if isinstance(category, str) else None,
            options=values,
        )


def _select_value(raw: Any, group: str | None = None) -> ACPConfigOptionValue | None:
    """One ACP ``SessionConfigSelectOption``; ``None`` without a string value."""
    value = getattr(raw, "value", None)
    if not isinstance(value, str):
        return None
    name = getattr(raw, "name", None)
    description = getattr(raw, "description", None)
    return ACPConfigOptionValue(
        value=value,
        name=name if isinstance(name, str) else value,
        description=description if isinstance(description, str) else None,
        group=group,
    )


def _flatten_select_values(entries: Sequence[Any]) -> list[ACPConfigOptionValue]:
    """One flat list from a select's options, whether grouped or not."""
    values: list[ACPConfigOptionValue | None] = []
    for entry in entries:
        grouped = getattr(entry, "options", None)
        if grouped is None:
            values.append(_select_value(entry))
            continue
        label = getattr(entry, "name", None)
        values.extend(
            _select_value(item, label if isinstance(label, str) else None)
            for item in grouped
        )
    return [value for value in values if value is not None]


class ACPSessionControls(BaseModel):
    """The slash commands and config options an ACP session offers now."""

    available_commands: list[ACPAvailableCommand] = Field(
        default_factory=list,
        description="The agent's slash commands, in the agent's order.",
    )
    config_options: list[ACPConfigOption] = Field(
        default_factory=list,
        description="The agent's session config options, in the agent's order.",
    )

    @classmethod
    def parse_commands(cls, raw: Sequence[Any]) -> list[ACPAvailableCommand]:
        """Normalize ACP commands, dropping unusable entries."""
        commands = (ACPAvailableCommand.from_protocol(item) for item in raw)
        return [command for command in commands if command is not None]

    @classmethod
    def parse_config_options(cls, raw: Sequence[Any]) -> list[ACPConfigOption]:
        """Normalize ACP config options, dropping unknown types."""
        options = (ACPConfigOption.from_protocol(item) for item in raw)
        return [option for option in options if option is not None]
