"""Live checks of ACP session controls against real ACP agents.

Two kinds, both deselected by default with the ``acp_live`` marker:

- **Built-in providers**, launched through ``npx`` with a bogus key as
  ``test_acp_conformance.py`` does: a preview succeeds and reports usable
  controls. No turn is sent, so no credential is needed.
- **Any agent named by the environment**, for agents outside the registry:

  - ``OPENHANDS_ACP_LIVE_AGENT_COMMAND``: the command, shell-split.
  - ``OPENHANDS_ACP_LIVE_CONFIG_OPTIONS``: a JSON object of start values.
  - ``OPENHANDS_ACP_LIVE_EXPECT_COMMANDS_CLEARED=1``: the agent clears its
    commands when it accepts the first prompt.

  These assert that a preview lists what the started session lists, and that
  the chosen values are the ones the session reports after its first prompt.
  Skipped when the command is unset.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
from collections.abc import Iterator
from pathlib import Path

import pytest

from openhands.sdk.agent.acp_agent import ACPAgent
from openhands.sdk.agent.acp_models import ACPSessionControls
from openhands.sdk.conversation import LocalConversation
from openhands.sdk.conversation.acp_preview import (
    PREVIEW_COMMANDS_WAIT_SECONDS,
    preview_acp_session,
)
from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.event import ACPSessionControlsEvent, MessageEvent
from openhands.sdk.settings.acp_providers import ACP_PROVIDERS
from openhands.sdk.workspace import LocalWorkspace
from tests.conftest import controls_events, wait_until
from tests.sdk.agent.test_acp_conformance import (
    _isolate_env,
    _skip_if_node_below_floor,
    requires_npx,
)


pytestmark = pytest.mark.acp_live

LIVE_AGENT_COMMAND = os.environ.get("OPENHANDS_ACP_LIVE_AGENT_COMMAND", "")
LIVE_CONFIG_OPTIONS: dict[str, str | bool] = json.loads(
    os.environ.get("OPENHANDS_ACP_LIVE_CONFIG_OPTIONS") or "{}"
)
EXPECT_COMMANDS_CLEARED = (
    os.environ.get("OPENHANDS_ACP_LIVE_EXPECT_COMMANDS_CLEARED") == "1"
)
FIRST_PROMPT = "Reply with the single word: ready."

requires_live_agent = pytest.mark.skipif(
    not LIVE_AGENT_COMMAND, reason="OPENHANDS_ACP_LIVE_AGENT_COMMAND is not set"
)


def settled_controls(conv: LocalConversation) -> ACPSessionControls:
    """The session's controls once the newest one is persisted."""
    agent = conv.agent
    assert isinstance(agent, ACPAgent)
    current = agent.wait_for_available_commands(PREVIEW_COMMANDS_WAIT_SECONDS)
    wait_until(
        lambda: bool(controls_events(conv.state.events))
        and controls_events(conv.state.events)[-1].controls == agent.session_controls,
        timeout=30,
    )
    return current


@pytest.fixture
def workspace(tmp_path: Path) -> LocalWorkspace:
    path = tmp_path / "workspace"
    path.mkdir()
    return LocalWorkspace(working_dir=str(path))


@pytest.fixture
def live_conversation(
    tmp_path: Path, workspace: LocalWorkspace
) -> Iterator[LocalConversation]:
    conv = LocalConversation(
        ACPAgent(
            acp_command=shlex.split(LIVE_AGENT_COMMAND),
            acp_config_options=LIVE_CONFIG_OPTIONS,
        ),
        workspace=workspace,
        persistence_dir=str(tmp_path / "conversations"),
        visualizer=None,
    )
    yield conv
    conv.close()


@requires_npx
@pytest.mark.parametrize("provider_key", list(ACP_PROVIDERS))
def test_a_built_in_provider_can_be_previewed(
    provider_key: str,
    tmp_path: Path,
    workspace: LocalWorkspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ACP_PROVIDERS[provider_key]
    _skip_if_node_below_floor(provider_key)
    _isolate_env(monkeypatch, tmp_path)
    agent = ACPAgent(
        acp_command=list(provider.default_command),
        acp_server=provider.key,
        acp_isolate_data_dir=True,
    )

    controls = preview_acp_session(agent, workspace, tmp_path / "preview")

    print(f"[acp-session-controls] provider={provider.key} {controls}")
    assert all(command.name for command in controls.available_commands)
    if agent._model_via_config_option:
        assert "model" in [option.id for option in controls.config_options]


@requires_live_agent
def test_the_preview_lists_what_the_started_session_lists(
    tmp_path: Path, workspace: LocalWorkspace, live_conversation: LocalConversation
) -> None:
    previewed = preview_acp_session(
        ACPAgent(
            acp_command=shlex.split(LIVE_AGENT_COMMAND),
            acp_config_options=LIVE_CONFIG_OPTIONS,
        ),
        workspace,
        tmp_path / "preview",
    )

    # No message: the session starts and the run ends before any prompt.
    live_conversation.run()

    assert previewed == settled_controls(live_conversation)


@requires_live_agent
def test_the_first_prompt_runs_with_the_chosen_values(
    live_conversation: LocalConversation,
) -> None:
    live_conversation.send_message(FIRST_PROMPT)

    # The agent may need a model key for the turn itself; what is asserted
    # here is what it reported once it accepted the prompt.
    with contextlib.suppress(ConversationRunError):
        live_conversation.run()

    reported = settled_controls(live_conversation)
    current = {option.id: option.current_value for option in reported.config_options}
    assert {k: current.get(k) for k in LIVE_CONFIG_OPTIONS} == LIVE_CONFIG_OPTIONS
    if EXPECT_COMMANDS_CLEARED:
        events = list(live_conversation.state.events)
        prompt_at = next(
            i
            for i, e in enumerate(events)
            if isinstance(e, MessageEvent) and e.source == "user"
        )
        after_prompt = [
            e for e in events[prompt_at:] if isinstance(e, ACPSessionControlsEvent)
        ]
        assert after_prompt, "the agent reported nothing after the first prompt"
        assert after_prompt[-1].available_commands == []
