"""ACP's unstable sub-agent types, carried until agent-client-protocol parses them.

ACP schema 1.24.1 added ``clientCapabilities.subagents`` and the ``subagent_update``,
``session_message`` and ``session_message_chunk`` session updates behind its unstable
flag; agent-client-protocol 0.12.1 drops all four. The models are upstream's generator
output for schema 1.24.1 (python-sdk 9d07d78), adapted to 0.12.1's base model. Delete
this module when ``test_acp_library_rejects_subagent_update`` fails.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Annotated, Any, Final, Literal

from acp.client.connection import ClientSideConnection
from acp.connection import JsonValue, MethodHandler
from acp.interfaces import Client
from acp.meta import AGENT_METHODS, CLIENT_METHODS
from acp.schema import (
    AudioContentBlock,
    BaseModel as ACPModel,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    Implementation,
    InitializeResponse,
    ResourceContentBlock,
    StopReason,
    TextContentBlock,
)
from acp.utils import request_model
from pydantic import ConfigDict, Field, TypeAdapter, ValidationError

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

UNSTABLE_SESSION_UPDATES: Final[frozenset[str]] = frozenset(
    {"subagent_update", "session_message", "session_message_chunk"}
)

ContentBlock = Annotated[
    TextContentBlock
    | ImageContentBlock
    | AudioContentBlock
    | ResourceContentBlock
    | EmbeddedResourceContentBlock,
    Field(discriminator="type"),
]


class SubagentCapabilities(ACPModel):
    field_meta: Annotated[dict[str, Any] | None, Field(alias="_meta")] = None


class SessionCancelCapabilities(ACPModel):
    field_meta: Annotated[dict[str, Any] | None, Field(alias="_meta")] = None


class SubagentSessionCapabilities(ACPModel):
    cancel: SessionCancelCapabilities | None = None
    field_meta: Annotated[dict[str, Any] | None, Field(alias="_meta")] = None


class SubagentState(ACPModel):
    """running, idle, requires_action, unknown, or a custom state kept whole."""

    model_config = ConfigDict(extra="allow")

    state: str
    stop_reason: Annotated[StopReason | None, Field(alias="stopReason")] = None
    field_meta: Annotated[dict[str, Any] | None, Field(alias="_meta")] = None


class SubagentUpdate(ACPModel):
    session_update: Annotated[
        Literal["subagent_update"],
        Field(alias="sessionUpdate"),
    ]
    session_id: Annotated[str, Field(alias="sessionId")]
    title: str | None = None
    description: str | None = None
    capabilities: SubagentSessionCapabilities | None = None
    state: SubagentState | None = None
    field_meta: Annotated[dict[str, Any] | None, Field(alias="_meta")] = None


class SessionMessage(ACPModel):
    session_update: Annotated[
        Literal["session_message"],
        Field(alias="sessionUpdate"),
    ]
    message_id: Annotated[str, Field(alias="messageId")]
    sender_session_id: Annotated[
        str | None,
        Field(alias="senderSessionId"),
    ] = None
    recipient_session_id: Annotated[
        str | None,
        Field(alias="recipientSessionId"),
    ] = None
    content: list[ContentBlock] | None = None
    field_meta: Annotated[dict[str, Any] | None, Field(alias="_meta")] = None


class SessionMessageChunk(ACPModel):
    session_update: Annotated[
        Literal["session_message_chunk"],
        Field(alias="sessionUpdate"),
    ]
    message_id: Annotated[str, Field(alias="messageId")]
    sender_session_id: Annotated[
        str | None,
        Field(alias="senderSessionId"),
    ] = None
    recipient_session_id: Annotated[
        str | None,
        Field(alias="recipientSessionId"),
    ] = None
    content: ContentBlock
    field_meta: Annotated[dict[str, Any] | None, Field(alias="_meta")] = None


UnstableSessionUpdate = SubagentUpdate | SessionMessage | SessionMessageChunk
UnstableUpdateHandler = Callable[[str, UnstableSessionUpdate], None]

_UNSTABLE_UPDATE_ADAPTER: Final = TypeAdapter(
    Annotated[UnstableSessionUpdate, Field(discriminator="session_update")]
)


class SubagentClientCapabilities(ClientCapabilities):
    subagents: SubagentCapabilities | None = None


SUBAGENT_CLIENT_CAPABILITIES: Final = SubagentClientCapabilities(
    subagents=SubagentCapabilities()
)


class _SubagentInitializeRequest(ACPModel):
    protocol_version: Annotated[int, Field(alias="protocolVersion")]
    client_capabilities: Annotated[
        SubagentClientCapabilities,
        Field(alias="clientCapabilities"),
    ]
    client_info: Annotated[Implementation | None, Field(alias="clientInfo")] = None
    field_meta: Annotated[dict[str, Any] | None, Field(alias="_meta")] = None


# ClientSideConnection is @final in agent-client-protocol 0.12.1; this subclass
# lives only until the library parses ACP's sub-agent updates itself.
class SubagentClientSideConnection(
    ClientSideConnection,  # pyright: ignore[reportGeneralTypeIssues]
):
    """A ClientSideConnection that hands ACP's unstable sub-agent updates to a
    callback ahead of the library's router, and can advertise ``subagents``."""

    def __init__(
        self,
        to_client: Client,
        input_stream: asyncio.StreamWriter,
        output_stream: asyncio.StreamReader,
        *,
        on_unstable_update: UnstableUpdateHandler,
    ) -> None:
        super().__init__(to_client, input_stream, output_stream)
        self._conn._handler = route_unstable_updates(
            self._conn._handler, on_unstable_update
        )

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        """Send ``initialize`` typed with ``SubagentClientCapabilities`` when given
        one, so ``subagents`` reaches the wire; otherwise the library's call."""
        if not isinstance(client_capabilities, SubagentClientCapabilities):
            return await super().initialize(
                protocol_version, client_capabilities, client_info, **kwargs
            )
        return await request_model(
            self._conn,
            AGENT_METHODS["initialize"],
            _SubagentInitializeRequest(
                protocol_version=protocol_version,
                client_capabilities=client_capabilities,
                client_info=client_info,
                field_meta=kwargs or None,
            ),
            InitializeResponse,
        )


def route_unstable_updates(
    inner: MethodHandler,
    on_unstable_update: UnstableUpdateHandler,
) -> MethodHandler:
    """Wrap a connection handler: the three unstable updates reach
    ``on_unstable_update`` synchronously, in arrival order; an invalid one is
    logged and dropped; every other message goes to ``inner``."""

    async def handler(
        method: str,
        params: JsonValue | None,
        is_notification: bool,
    ) -> JsonValue | None:
        kind = _unstable_update_kind(method, params, is_notification)
        if kind is None:
            return await inner(method, params, is_notification)
        assert isinstance(params, dict)
        session_id = params.get("sessionId")
        try:
            update = _UNSTABLE_UPDATE_ADAPTER.validate_python(params["update"])
        except ValidationError as error:
            logger.warning(
                "Dropping an invalid ACP %s (%d validation errors)",
                kind,
                error.error_count(),
            )
            return None
        if not isinstance(session_id, str):
            logger.warning("Dropping an ACP %s without a sessionId", kind)
            return None
        on_unstable_update(session_id, update)
        return None

    return handler


def _unstable_update_kind(
    method: str,
    params: JsonValue | None,
    is_notification: bool,
) -> str | None:
    """The ``sessionUpdate`` of an unstable session/update notification, else None."""
    if not is_notification or method != CLIENT_METHODS["session_update"]:
        return None
    if not isinstance(params, dict) or not isinstance(params.get("update"), dict):
        return None
    kind = params["update"].get("sessionUpdate")
    return kind if kind in UNSTABLE_SESSION_UPDATES else None
