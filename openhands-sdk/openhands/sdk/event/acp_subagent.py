"""Sub-agent sessions of an ACP agent, as persisted events.

ACP schema 1.24.1 (unstable) lets an agent expose sub-agent sessions: a child
session announced with ``subagent_update``, messages between sessions, and the
child's own text. Each kind here is stored only for agents with
``ACPAgent.acp_subagents``.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field
from rich.text import Text

from openhands.sdk.event.base import Event
from openhands.sdk.event.types import SourceType


_MAX_DISPLAY_CHARS = 500


def _clip(text: str) -> str:
    if len(text) <= _MAX_DISPLAY_CHARS:
        return text
    return text[:_MAX_DISPLAY_CHARS] + "..."


class ACPSubagentEvent(Event):
    """An ACP sub-agent session's association with its parent, as last reported.

    ACP's ``subagent_update`` (schema 1.24.1, unstable) with its patch semantics
    already applied: each event is the whole current association, so consumers
    keep the latest per ``acp_session_id``. Written per ``subagent_update``, per
    change of the child's reported cost, and, with ``source="environment"``, when
    a new ACP connection starts (state unconfirmed, cancel withdrawn).
    """

    source: SourceType = "agent"
    acp_session_id: str = Field(description="The child's ACP session id.")
    parent_session_id: str | None = Field(
        default=None,
        description="The parent's ACP session id; None for the root session.",
    )
    parent_tool_call_id: str | None = Field(
        default=None,
        description=(
            "The parent's tool call that spawned the child, from "
            "_meta.openhands.parentToolCallId."
        ),
    )
    title: str | None = None
    description: str | None = None
    state: str | None = Field(
        default=None,
        description=(
            "'running', 'idle', 'requires_action', 'unknown', an agent-specific "
            "value, or None when the current state is unconfirmed."
        ),
    )
    stop_reason: str | None = None
    cancellable: bool = Field(
        default=False,
        description="Whether a client may cancel this child's work now.",
    )
    cost: float | None = Field(
        default=None,
        description="The child's latest cumulative cost; never add it to others.",
    )
    cost_currency: str | None = None
    meta: dict[str, Any] | None = Field(
        default=None,
        description="The association's ACP _meta, verbatim.",
    )

    @property
    def visualize(self) -> Text:
        """The child, its parent and cell, its state and its cost."""
        content = Text()
        content.append(self.title or self.acp_session_id, style="bold")
        parent = self.parent_session_id or "root"
        cell = f" in {self.parent_tool_call_id}" if self.parent_tool_call_id else ""
        content.append(f"\nparent={parent}{cell}", style="dim")
        state = self.state or "unconfirmed"
        if self.stop_reason:
            state = f"{state} ({self.stop_reason})"
        content.append(f"\nstate={state}")
        if self.cost is not None:
            content.append(f" | cost={self.cost:g} {self.cost_currency or ''}".rstrip())
        return content

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__} ({self.source}): {self.acp_session_id} "
            f"[{self.state or 'unconfirmed'}]"
        )


class ACPSessionMessageEvent(Event):
    """A message between ACP sessions, as one session's transcript shows it.

    ACP's ``session_message`` and accumulated ``session_message_chunk`` (schema
    1.24.1, unstable). Upserts: consumers keep the latest per
    ``(acp_session_id, message_id)``.
    """

    source: SourceType = "agent"
    acp_session_id: str | None = Field(
        default=None,
        description="The transcript this entry belongs to; None for the root.",
    )
    message_id: str
    sender_session_id: str | None = None
    recipient_session_id: str | None = None
    text: str = ""
    meta: dict[str, Any] | None = None

    @property
    def visualize(self) -> Text:
        """Sender and recipient, then the text."""
        content = Text()
        sender = self.sender_session_id or "?"
        recipient = self.recipient_session_id or "?"
        content.append(f"{sender} -> {recipient}", style="dim")
        content.append(f"\n{_clip(self.text)}")
        return content

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__} ({self.source}): "
            f"{self.sender_session_id} -> {self.recipient_session_id}"
        )


class ACPSessionTextEvent(Event):
    """A run of a sub-agent session's own streamed text or reasoning.

    Consecutive ``agent_message_chunk`` (``thought`` false) or
    ``agent_thought_chunk`` (``thought`` true) updates of one child session,
    stored once the run ends.
    """

    source: SourceType = "agent"
    acp_session_id: str
    thought: bool = False
    text: str

    @property
    def visualize(self) -> Text:
        """The session, then the text; reasoning is dimmed."""
        content = Text()
        content.append(self.acp_session_id, style="bold")
        content.append(f"\n{_clip(self.text)}", style="dim" if self.thought else "")
        return content

    def __str__(self) -> str:
        kind = "thought" if self.thought else "text"
        return (
            f"{self.__class__.__name__} ({self.source}): {self.acp_session_id} {kind}"
        )
