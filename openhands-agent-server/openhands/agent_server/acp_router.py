"""ACP session controls: preview what an agent offers, and set its options."""

from typing import Final
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from openhands.agent_server.conversation_service import (
    ConversationService,
    InvalidACPConfigOptions,
)
from openhands.agent_server.dependencies import get_conversation_service
from openhands.agent_server.models import StartConversationRequest
from openhands.sdk.agent.acp_agent import ACPConfigOptionRejectedError
from openhands.sdk.agent.acp_models import ACPSessionControls
from openhands.sdk.conversation.acp_preview import ACPPreviewError
from openhands.sdk.profiles.resolver import DanglingMcpServerRef, ProfileNotFound


acp_router = APIRouter(prefix="/acp", tags=["ACP"])
conversation_acp_router = APIRouter(
    prefix="/conversations/{conversation_id}/acp",
    tags=["ACP"],
)

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
        min_length=1,
        description="The option's id, as the agent reports it.",
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
    except ProfileNotFound as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except DanglingMcpServerRef as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": str(e), "dangling_mcp_server_refs": e.missing},
        ) from e
    except InvalidACPConfigOptions as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        ) from e
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
        applied=controls is not None,
        controls=controls or ACPSessionControls(),
    )
