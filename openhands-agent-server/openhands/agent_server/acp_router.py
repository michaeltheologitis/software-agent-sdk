"""ACP routes: the models an ACP server offers and a preview of its session
controls before a conversation starts, setting its options, and cancelling one
of its sub-agent sessions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Final
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from openhands.agent_server._secrets_exposure import get_config
from openhands.agent_server.config import Config
from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.credential_binding import LocalVersionedCredentialBinding
from openhands.agent_server.dependencies import get_conversation_service
from openhands.agent_server.launch import launch_http_exception
from openhands.agent_server.models import StartConversationRequest
from openhands.agent_server.persistence import get_secrets_store
from openhands.sdk.agent.acp_agent import ACPConfigOptionRejectedError
from openhands.sdk.agent.acp_file_credentials import CODEX_AUTH_SECRET_NAME
from openhands.sdk.agent.acp_model_discovery import discover_acp_models
from openhands.sdk.agent.acp_models import ACPModelDiscovery, ACPSessionControls
from openhands.sdk.agent.acp_subagents import (
    ACPSessionNotCancellableError,
    ACPSessionNotFoundError,
)
from openhands.sdk.conversation.acp_preview import ACPPreviewError
from openhands.sdk.credential import VersionedCredentialBinding
from openhands.sdk.launch import AgentLaunchError, LaunchStoreError
from openhands.sdk.logger import get_logger
from openhands.sdk.profiles.resolver import ProfileNotFound
from openhands.sdk.secret import SecretSource
from openhands.sdk.settings.model import ACPAgentSettings


logger = get_logger(__name__)

acp_router = APIRouter(prefix="/acp", tags=["ACP"])
conversation_acp_router = APIRouter(
    prefix="/conversations/{conversation_id}/acp", tags=["ACP"]
)

_CACHE_TTL_SECONDS = 300.0
_MAX_CONCURRENT_DISCOVERIES = 2


class ACPModelDiscoveryRequest(BaseModel):
    agent_settings: ACPAgentSettings = Field(
        description=(
            "The ACP agent to ask. Only the launch fields are used; "
            "``acp_model``, MCP servers and agent context are ignored."
        ),
    )
    secrets: dict[str, SecretSource] = Field(
        default_factory=dict,
        description=(
            "Credentials for the server, in the same shape as a conversation "
            "start request's ``secrets`` (typically ``LookupSecret`` references "
            "to this server's stored secrets)."
        ),
    )
    refresh: bool = Field(
        default=False, description="Ask the server again instead of reusing a result."
    )


@dataclass
class _DiscoveryCache:
    results: dict[str, tuple[float, ACPModelDiscovery]] = field(default_factory=dict)
    in_flight: dict[str, asyncio.Future[ACPModelDiscovery]] = field(
        default_factory=dict
    )
    slots: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(_MAX_CONCURRENT_DISCOVERIES)
    )

    async def get(
        self,
        key: str,
        discover: Callable[[], Awaitable[ACPModelDiscovery]],
        *,
        refresh: bool,
    ) -> ACPModelDiscovery:
        cached = self.results.get(key)
        if not refresh and cached is not None and cached[0] > time.monotonic():
            return cached[1]
        pending = self.in_flight.get(key)
        if pending is None:
            pending = asyncio.ensure_future(self._run(key, discover))
            self.in_flight[key] = pending
        return await asyncio.shield(pending)

    async def _run(
        self, key: str, discover: Callable[[], Awaitable[ACPModelDiscovery]]
    ) -> ACPModelDiscovery:
        try:
            async with self.slots:
                result = await discover()
            if result.error is None:
                self.results[key] = (time.monotonic() + _CACHE_TTL_SECONDS, result)
            return result
        finally:
            self.in_flight.pop(key, None)


def _discovery_cache(request: Request) -> _DiscoveryCache:
    cache = getattr(request.app.state, "acp_model_discovery_cache", None)
    if cache is None:
        cache = _DiscoveryCache()
        request.app.state.acp_model_discovery_cache = cache
    return cache


def _resolve_secrets(secrets: Mapping[str, SecretSource]) -> dict[str, str]:
    values: dict[str, str] = {}
    for name, source in secrets.items():
        try:
            value = source.get_value()
        except Exception:
            logger.warning("Could not resolve secret %r for ACP model discovery", name)
            continue
        if value is not None:
            values[name] = value
    return values


def _cache_key(
    settings: ACPAgentSettings, command: list[str], secrets: Mapping[str, str]
) -> str:
    payload = {
        "command": command,
        "args": list(settings.acp_args),
        "session_mode": settings.acp_session_mode,
        "isolate": settings.acp_isolate_data_dir,
        "file_secrets": [
            spec.model_dump(mode="json") for spec in settings.acp_file_secrets
        ],
        "secrets": sorted(secrets.items()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _codex_bindings(
    settings: ACPAgentSettings, config: Config, secrets: dict[str, str]
) -> dict[str, VersionedCredentialBinding]:
    if settings.acp_server != "codex":
        return {}
    store = get_secrets_store(config)
    if store.get_secret(CODEX_AUTH_SECRET_NAME) is None:
        return {}
    # A stored ChatGPT login refreshes through the store, as in a conversation.
    secrets.pop(CODEX_AUTH_SECRET_NAME, None)
    return {
        CODEX_AUTH_SECRET_NAME: LocalVersionedCredentialBinding(
            store, CODEX_AUTH_SECRET_NAME
        )
    }


@acp_router.post("/models", response_model=ACPModelDiscovery)
async def discover_models(
    body: ACPModelDiscoveryRequest, request: Request
) -> ACPModelDiscovery:
    """Start the ACP server in a throwaway session and report its models.

    The server answers with its default model and the models it offers for the
    given credentials; a launch or authentication failure comes back in
    ``error`` rather than as an HTTP error.
    """
    config: Config = get_config(request)
    if config.conversation_runtime == "docker":
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            "ACP model discovery runs only where conversations run in-process",
        )
    settings = body.agent_settings
    try:
        command = settings.resolve_acp_command()
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    secrets = await asyncio.to_thread(_resolve_secrets, body.secrets)
    bindings = await asyncio.to_thread(_codex_bindings, settings, config, secrets)
    key = _cache_key(settings, command, secrets)

    async def run() -> ACPModelDiscovery:
        config.conversations_path.mkdir(parents=True, exist_ok=True)
        return await asyncio.to_thread(
            discover_acp_models,
            settings,
            secrets=secrets,
            credential_bindings=bindings,
            # Shares the conversations' npx cache.
            work_root=config.conversations_path,
        )

    return await _discovery_cache(request).get(key, run, refresh=body.refresh)


# Never 401 for an agent's authentication failure: clients read 401 as their
# own session to the agent-server having expired.
_PREVIEW_ERROR_STATUS: Final[dict[str, int]] = {
    "ACPConfigOptionRejected": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "ACPStartupTimeout": status.HTTP_504_GATEWAY_TIMEOUT,
}
_PREVIEW_ERROR_DEFAULT_STATUS: Final[int] = status.HTTP_502_BAD_GATEWAY


class ACPConfigOptionSetRequest(BaseModel):
    """Set one ACP session config option."""

    config_id: str = Field(
        min_length=1, description="The option's id, as the agent reports it."
    )
    value: str | bool = Field(
        description="A select option's value, or a boolean option's value.",
    )


class ACPConfigOptionSetResponse(BaseModel):
    """What setting an option did."""

    applied: bool = Field(
        description=(
            "True when a live session took the value; False when it is kept for "
            "the session's start."
        ),
    )
    controls: ACPSessionControls = Field(
        description="The session's controls after the set; empty when not applied.",
    )


@acp_router.post(
    "/preview",
    responses={
        400: {"description": "The resolved agent is not an ACP agent"},
        404: {"description": "Agent profile not found"},
        422: {"description": "Invalid request, or the agent refused an option value"},
        429: {"description": "Conversation run limit reached"},
        501: {"description": "Unavailable in Docker runtime mode"},
        502: {"description": "The agent failed to start"},
        504: {"description": "The agent did not start in time"},
    },
)
async def preview_acp_session(
    request: StartConversationRequest,
    http_request: Request,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ACPSessionControls:
    """What an ACP agent would offer for this start request, before it exists.

    The body is the payload a client sends to start a conversation, with the
    chosen ``acp_config_options``. The agent is started once, in a throwaway
    session, and stopped again.
    """
    # The Docker runtime runs agents in conversation containers; a preview on
    # this server would run the agent outside them.
    if http_request.app.state.config.conversation_runtime == "docker":
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            "This operation is unavailable in Docker runtime mode",
        )
    try:
        return await conversation_service.preview_acp_session(request)
    except (ProfileNotFound, AgentLaunchError, LaunchStoreError) as e:
        raise launch_http_exception(e) from e
    except ACPPreviewError as e:
        raise HTTPException(
            status_code=_PREVIEW_ERROR_STATUS.get(
                e.code, _PREVIEW_ERROR_DEFAULT_STATUS
            ),
            detail=e.detail,
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e


@conversation_acp_router.post(
    "/config-options",
    responses={
        400: {"description": "Not an ACP conversation, or the model option"},
        404: {"description": "Conversation not found"},
        422: {"description": "The agent refused the value"},
        504: {"description": "The agent did not answer in time"},
    },
)
async def set_acp_config_option(
    conversation_id: UUID,
    request: ACPConfigOptionSetRequest,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ACPConfigOptionSetResponse:
    """Set an ACP session config option, live or for the session's start.

    On a live session the agent answers with its controls, which may change
    other options too. Before the session starts, the value is kept and applied
    after ``session/new``. The model option is set with ``switch_acp_model``.
    """
    event_service = await conversation_service.get_event_service(conversation_id)
    if event_service is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    try:
        controls = await event_service.set_acp_config_option(
            request.config_id, request.value
        )
    except ACPConfigOptionRejectedError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e
    except TimeoutError as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(e)
        ) from e
    return ACPConfigOptionSetResponse(
        applied=controls is not None, controls=controls or ACPSessionControls()
    )


class CancelACPSessionResponse(BaseModel):
    """A cancel sent to an ACP sub-agent session; its outcome arrives later."""

    session_id: str = Field(description="The ACP session the cancel was sent for.")
    requested: bool = Field(
        default=True,
        description="Always true; the child's next state update confirms it.",
    )


@conversation_acp_router.post(
    "/sessions/{session_id}/cancel",
    responses={
        400: {"description": "The conversation's agent is not an ACP agent"},
        404: {"description": "Conversation or ACP sub-agent session not found"},
        409: {"description": "The ACP session does not accept cancel right now"},
        504: {"description": "The ACP server did not take the cancel in time"},
    },
)
async def cancel_conversation_acp_session(
    conversation_id: UUID,
    session_id: str,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> CancelACPSessionResponse:
    """Cancel the current work of one ACP sub-agent session.

    Sends ``session/cancel`` for a child session the ACP agent announced with a
    ``cancel`` capability. The child's next ``ACPSubagentEvent`` (idle, stop
    reason ``cancelled``) confirms it.
    """
    event_service = await conversation_service.get_event_service(conversation_id)
    if event_service is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    try:
        await event_service.cancel_acp_session(session_id)
    except ACPSessionNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"ACP session {session_id} is not a sub-agent session of this "
                "conversation."
            ),
        ) from e
    except ACPSessionNotCancellableError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"ACP session {session_id} does not accept cancel; cancel the "
                "conversation's turn instead."
            ),
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e
    except TimeoutError as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(e)
        ) from e
    return CancelACPSessionResponse(session_id=session_id)
