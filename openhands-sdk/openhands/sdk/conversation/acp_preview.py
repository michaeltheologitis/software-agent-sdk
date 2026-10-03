"""What an ACP agent would offer, before any conversation exists.

A preview starts the agent exactly as a conversation's first run would, in a
throwaway :class:`ConversationState`, reads the slash commands and config
options the session reports, and closes it again.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from openhands.sdk.agent.acp_agent import (
    ACPAgent,
    _acp_error_detail,
    _classify_acp_init_error,
)
from openhands.sdk.agent.acp_models import ACPSessionControls
from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.secret import SecretValue
from openhands.sdk.utils.cipher import Cipher
from openhands.sdk.workspace import LocalWorkspace


PREVIEW_COMMANDS_WAIT_SECONDS: Final[float] = 2.0


class ACPPreviewError(RuntimeError):
    """The agent could not be previewed.

    ``code`` is the ConversationErrorEvent code a start would have reported;
    ``detail`` is redacted and masked.
    """

    def __init__(
        self,
        code: str,
        detail: str,
    ) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def preview_acp_session(
    agent: ACPAgent,
    workspace: LocalWorkspace,
    persistence_dir: Path,
    *,
    secrets: Mapping[str, SecretValue] | None = None,
    cipher: Cipher | None = None,
    commands_wait_seconds: float = PREVIEW_COMMANDS_WAIT_SECONDS,
) -> ACPSessionControls:
    """Start ``agent``'s session in a throwaway state, read what it offers, close it.

    Runs ACPAgent.init_state, so the agent starts exactly as a conversation's
    would, ``agent.acp_config_options`` included. Blocking; the caller deletes
    ``persistence_dir``.

    Raises:
        ACPPreviewError: The agent failed to start or refused an option value.
    """
    persistence_dir.mkdir(parents=True, exist_ok=True)
    if not Path(workspace.working_dir).is_dir():
        # What the agent would see in a fresh folder, without creating anything
        # in the user's tree.
        scratch = persistence_dir / "workspace"
        scratch.mkdir(exist_ok=True)
        workspace = LocalWorkspace(working_dir=str(scratch))
    state = ConversationState.create(
        id=uuid.uuid4(),
        agent=agent,
        workspace=workspace,
        persistence_dir=str(persistence_dir),
        cipher=cipher,
    )
    # Seeded as LocalConversation seeds a new conversation: the agent
    # context's secrets, then the request's over them.
    if agent.agent_context is not None and agent.agent_context.secrets:
        state.secret_registry.update_secrets(agent.agent_context.secrets)
    if secrets:
        state.secret_registry.update_secrets(secrets)
    try:
        with state:
            try:
                agent.init_state(state, on_event=lambda _event: None)
            except Exception as exc:
                raise ACPPreviewError(
                    _classify_acp_init_error(exc),
                    _acp_error_detail(exc, state.secret_registry),
                ) from exc
        controls = agent.wait_for_available_commands(commands_wait_seconds)
        agent.close_acp_session()
        return controls
    finally:
        agent.close()
