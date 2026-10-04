"""Sub-agent sessions of an ACP agent, as the bridge sees them on one connection.

The client side of ACP's sub-agent sessions (schema 1.24.1, unstable): which sessions
are children, their merged association, their streamed text and directed messages,
their cost, and whether a client may cancel them now. Bookkeeping only: each method
returns the events to persist; nothing here emits, locks, awaits or does I/O.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from acp.schema import TextContentBlock, UsageUpdate

from openhands.sdk.agent.acp_unstable import (
    SessionMessage,
    SessionMessageChunk,
    SubagentUpdate,
)
from openhands.sdk.event import (
    ACPSessionMessageEvent,
    ACPSessionTextEvent,
    ACPSubagentEvent,
    Event,
)
from openhands.sdk.event.types import SourceType
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

# Each stored snapshot is a new event: its own id and time, and no parent yet.
_PER_EVENT_FIELDS = {"id", "timestamp", "parent_id"}


class ACPSessionNotFoundError(LookupError):
    """No sub-agent session with this id is known on the ACP connection."""


class ACPSessionNotCancellableError(RuntimeError):
    """The session cannot be cancelled now: no live connection, the root session,
    or no current ``cancel`` grant from the agent."""


@dataclass
class _Pending:
    thought: bool | None
    message_id: str | None
    parts: list[str] = field(default_factory=list)


@dataclass
class _Message:
    sender_session_id: str | None = None
    recipient_session_id: str | None = None
    text: str = ""
    meta: dict[str, Any] | None = None


class ACPSubagentSessions:
    """Routing, merge, segments, cost and cancel grants for one ACP connection.

    A child's association is held as its latest ``ACPSubagentEvent``, whose
    ``cancellable`` is the live grant; each snapshot stored is a new event.
    """

    root_session_id: str | None
    replaying: bool

    def __init__(self, *, mask: Callable[[Any], Any]) -> None:
        self._mask = mask
        self.root_session_id = None
        self.replaying = False
        self._children: dict[str, ACPSubagentEvent] = {}
        self._pending: dict[str, _Pending] = {}
        self._messages: dict[tuple[str, str], _Message] = {}

    def seed(self, events: Iterable[Event]) -> list[ACPSubagentEvent]:
        """Register the children stored in ``events``; return the snapshots that
        withdraw their cancel grants and unconfirm their states."""
        stored = {
            e.acp_session_id: e for e in events if isinstance(e, ACPSubagentEvent)
        }
        unconfirmed = {"state": None, "stop_reason": None, "cancellable": False}
        for child, snapshot in stored.items():
            self._children[child] = snapshot.model_copy(update=unconfirmed)
        return [
            self._snapshot(child, source="environment")
            for child, snapshot in stored.items()
            if snapshot.state not in (None, "idle")
        ]

    def is_child(self, session_id: str) -> bool:
        return session_id in self._children

    def key(self, session_id: str) -> str | None:
        """The stored form of a session id: ``None`` for the root."""
        return None if session_id == self.root_session_id else session_id

    def on_subagent_update(
        self, session_id: str, update: SubagentUpdate
    ) -> list[Event]:
        child = update.session_id
        if child in (self.root_session_id, session_id):
            logger.warning(
                "Ignoring an ACP subagent_update that names the root session or "
                "the session it arrived on"
            )
            return []
        parent = self.key(session_id)
        association = self._children.get(child)
        if association is None:
            self._children[child] = ACPSubagentEvent(
                acp_session_id=child, parent_session_id=parent
            )
        elif association.parent_session_id != parent:
            logger.warning(
                "An ACP subagent_update arrived on a session other than its "
                "child's parent; the child keeps its parent"
            )
        if self.replaying:
            return []
        events = self._flush(session_id) + self._flush(child)
        self._merge(child, update)
        return [*events, self._snapshot(child)]

    def on_session_message(
        self, session_id: str, update: SessionMessage
    ) -> list[Event]:
        if self.replaying:
            return []
        events = self._flush(session_id)
        message = self._message(session_id, update)
        fields = update.model_fields_set
        if "content" in fields:
            message.text = self._text_of(update.content or [])
        if "field_meta" in fields:
            message.meta = self._mask(update.field_meta)
        return [*events, self._message_event(session_id, update.message_id)]

    def on_session_message_chunk(
        self, session_id: str, update: SessionMessageChunk
    ) -> list[Event]:
        if self.replaying:
            return []
        pending = self._pending.get(session_id)
        events: list[Event] = []
        if pending is None or pending.message_id != update.message_id:
            events = self._flush(session_id)
            pending = self._pending[session_id] = _Pending(None, update.message_id)
        self._message(session_id, update)
        if isinstance(update.content, TextContentBlock):
            pending.parts.append(update.content.text)
        else:
            logger.debug(
                "Not storing a %s block of ACP message %s",
                update.content.type,
                update.message_id,
            )
        return events

    def on_child_text(
        self, session_id: str, text: str, *, thought: bool
    ) -> list[Event]:
        if self.replaying:
            return []
        pending = self._pending.get(session_id)
        events: list[Event] = []
        if (
            pending is None
            or pending.message_id is not None
            or (pending.thought != thought)
        ):
            events = self._flush(session_id)
            pending = self._pending[session_id] = _Pending(thought, None)
        pending.parts.append(text)
        return events

    def on_child_usage(self, session_id: str, update: UsageUpdate) -> list[Event]:
        association = self._children.get(session_id)
        if self.replaying or association is None:
            return []
        cost = update.cost.amount if update.cost is not None else None
        currency = update.cost.currency if update.cost is not None else None
        if (cost, currency) == (association.cost, association.cost_currency):
            return []
        self._children[session_id] = association.model_copy(
            update={"cost": cost, "cost_currency": currency}
        )
        return [self._snapshot(session_id)]

    def before_update(self, session_id: str) -> list[Event]:
        """Flush ``session_id``'s open segment, which any update other than
        text, a message chunk or usage ends."""
        return [] if self.replaying else self._flush(session_id)

    def flush_all(self) -> list[Event]:
        events: list[Event] = []
        for session_id in list(self._pending):
            events += self._flush(session_id)
        return events

    def check_cancel(self, session_id: str) -> None:
        """Raise unless the agent granted ``cancel`` for this child, live."""
        if session_id == self.root_session_id:
            raise ACPSessionNotCancellableError(session_id)
        association = self._children.get(session_id)
        if association is None:
            raise ACPSessionNotFoundError(session_id)
        if not association.cancellable:
            raise ACPSessionNotCancellableError(session_id)

    def _merge(self, child: str, update: SubagentUpdate) -> None:
        """Apply ACP's patch rules: omitted keeps, null clears, a value replaces."""
        fields = update.model_fields_set
        changes: dict[str, Any] = {}
        if "title" in fields:
            changes["title"] = self._mask(update.title)
        if "description" in fields:
            changes["description"] = self._mask(update.description)
        if "state" in fields:
            state = update.state
            changes["state"] = state.state if state is not None else None
            changes["stop_reason"] = state.stop_reason if state is not None else None
        if "capabilities" in fields:
            capabilities = update.capabilities
            changes["cancellable"] = (
                capabilities is not None and capabilities.cancel is not None
            )
        if "field_meta" in fields:
            meta = update.field_meta
            changes["meta"] = self._mask(meta)
            parent_tool_call_id = _parent_tool_call_id(meta)
            if parent_tool_call_id is not None:
                changes["parent_tool_call_id"] = parent_tool_call_id
        self._children[child] = self._children[child].model_copy(update=changes)

    def _snapshot(
        self, child: str, *, source: SourceType = "agent"
    ) -> ACPSubagentEvent:
        association = self._children[child].model_dump(exclude=_PER_EVENT_FIELDS)
        return ACPSubagentEvent.model_validate({**association, "source": source})

    def _message(
        self, session_id: str, update: SessionMessage | SessionMessageChunk
    ) -> _Message:
        """The resolved message, with any participants the update names."""
        message = self._messages.setdefault((session_id, update.message_id), _Message())
        if update.sender_session_id is not None:
            message.sender_session_id = update.sender_session_id
        if update.recipient_session_id is not None:
            message.recipient_session_id = update.recipient_session_id
        return message

    def _message_event(
        self, session_id: str, message_id: str
    ) -> ACPSessionMessageEvent:
        message = self._messages[(session_id, message_id)]
        return ACPSessionMessageEvent(
            acp_session_id=self.key(session_id),
            message_id=message_id,
            sender_session_id=message.sender_session_id,
            recipient_session_id=message.recipient_session_id,
            text=message.text,
            meta=message.meta,
        )

    def _flush(self, session_id: str) -> list[Event]:
        pending = self._pending.pop(session_id, None)
        if pending is None or not pending.parts:
            return []
        text = self._mask("".join(pending.parts))
        if pending.message_id is None:
            return [
                ACPSessionTextEvent(
                    acp_session_id=session_id, thought=bool(pending.thought), text=text
                )
            ]
        self._messages[(session_id, pending.message_id)].text += text
        return [self._message_event(session_id, pending.message_id)]

    def _text_of(self, blocks: Sequence[Any]) -> str:
        texts = [b.text for b in blocks if isinstance(b, TextContentBlock)]
        if len(texts) < len(blocks):
            logger.debug(
                "Not storing %d non-text block(s) of an ACP message",
                len(blocks) - len(texts),
            )
        return self._mask("".join(texts))


def _parent_tool_call_id(meta: dict[str, Any] | None) -> str | None:
    """``_meta.openhands.parentToolCallId`` when it is a string."""
    openhands = (meta or {}).get("openhands")
    if not isinstance(openhands, dict):
        return None
    value = openhands.get("parentToolCallId")
    return value if isinstance(value, str) else None
