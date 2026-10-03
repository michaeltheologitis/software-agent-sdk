"""ACPSessionControlsEvent — the commands and options an ACP session offers."""

from __future__ import annotations

from pydantic import Field
from rich.text import Text

from openhands.sdk.agent.acp_models import (
    ACPAvailableCommand,
    ACPConfigOption,
    ACPSessionControls,
)
from openhands.sdk.event.base import Event
from openhands.sdk.event.types import SourceType


class ACPSessionControlsEvent(Event):
    """The slash commands and config options an ACP session offers now.

    Latest wins: every event carries both lists in full, and the newest event
    of a conversation is its current state. Emitted when the session starts
    and whenever the agent reports a change.
    """

    source: SourceType = "agent"
    available_commands: list[ACPAvailableCommand] = Field(
        default_factory=list, description="The agent's slash commands."
    )
    config_options: list[ACPConfigOption] = Field(
        default_factory=list, description="The agent's session config options."
    )

    @classmethod
    def from_controls(cls, controls: ACPSessionControls) -> ACPSessionControlsEvent:
        """Build the event for one snapshot."""
        return cls(
            available_commands=controls.available_commands,
            config_options=controls.config_options,
        )

    @property
    def controls(self) -> ACPSessionControls:
        """The snapshot this event carries."""
        return ACPSessionControls(
            available_commands=self.available_commands,
            config_options=self.config_options,
        )

    @property
    def visualize(self) -> Text:
        """One line: the command names, then each option's id and current value."""
        commands = " ".join(f"/{c.name}" for c in self.available_commands) or "none"
        options = (
            ", ".join(f"{o.id}={o.current_value}" for o in self.config_options)
            or "none"
        )
        return Text(f"Commands: {commands} | Options: {options}")

    def __str__(self) -> str:
        return f"{self.__class__.__name__} ({self.source}): {self.visualize.plain}"
