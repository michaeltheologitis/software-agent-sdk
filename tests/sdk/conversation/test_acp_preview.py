"""preview_acp_session: what an ACP agent offers before any conversation exists."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil
import pytest

from openhands.sdk.agent.acp_agent import ACPAgent
from openhands.sdk.agent.acp_models import ACPSessionControls
from openhands.sdk.conversation.acp_preview import (
    PREVIEW_COMMANDS_WAIT_SECONDS,
    ACPPreviewError,
    preview_acp_session,
)
from openhands.sdk.workspace import LocalWorkspace
from tests.conftest import (
    SCRIPTED_ACP_AGENT,
    controls_events,
    scripted_acp_command,
    wait_until,
)


def live_scripted_agents() -> list[psutil.Process]:
    return [
        child
        for child in psutil.Process().children(recursive=True)
        if child.status() != psutil.STATUS_ZOMBIE
        and str(SCRIPTED_ACP_AGENT) in " ".join(child.cmdline())
    ]


@pytest.fixture
def workspace(tmp_path: Path) -> LocalWorkspace:
    path = tmp_path / "workspace"
    path.mkdir()
    return LocalWorkspace(working_dir=str(path))


@pytest.fixture
def preview(
    tmp_path: Path, workspace: LocalWorkspace
) -> Callable[..., ACPSessionControls]:
    def _preview(*flags: str, **fields: Any) -> ACPSessionControls:
        agent = ACPAgent(acp_command=scripted_acp_command(*flags), **fields)
        return preview_acp_session(agent, workspace, tmp_path / "preview")

    return _preview


@pytest.mark.parametrize(
    "values, commands",
    [
        ({}, ["summarize"]),
        ({"profile": "fast"}, ["summarize"]),
        ({"profile": "thorough"}, ["summarize", "compare"]),
    ],
)
def test_the_preview_equals_the_started_session_before_its_first_prompt(
    preview, scripted_conversation, values, commands
):
    previewed = preview(acp_config_options=values)
    conv = scripted_conversation(acp_config_options=values)

    # No message: the session starts, and the run ends without a prompt.
    conv.run()

    wait_until(
        lambda: any(
            e.available_commands for e in controls_events(conv.state.events)[-1:]
        )
    )
    assert [c.name for c in previewed.available_commands] == commands
    assert previewed == controls_events(conv.state.events)[-1].controls


def test_session_close_is_sent_when_the_agent_advertises_it(preview, acp_request_log):
    preview()

    assert [e["method"] for e in acp_request_log()][-1] == "session/close"


def test_session_close_is_not_sent_when_the_agent_does_not_advertise_it(
    preview, acp_request_log
):
    preview("--no-close")

    assert "session/close" not in [e["method"] for e in acp_request_log()]


def test_an_agent_that_never_reports_commands_is_previewed_after_the_wait(preview):
    began = time.monotonic()

    previewed = preview("--no-commands")

    elapsed = time.monotonic() - began
    assert PREVIEW_COMMANDS_WAIT_SECONDS <= elapsed < PREVIEW_COMMANDS_WAIT_SECONDS + 10
    assert previewed.available_commands == []
    assert [o.id for o in previewed.config_options] == ["profile"]


@pytest.mark.parametrize(
    "fields",
    [{}, {"acp_config_options": {"profile": "turbo"}}],
    ids=["previewed", "refused"],
)
def test_the_agent_process_is_gone_afterwards(preview, fields):
    with pytest.raises(ACPPreviewError) if fields else contextlib.nullcontext():
        preview(**fields)

    assert live_scripted_agents() == []


def test_a_missing_working_directory_is_previewed_from_an_empty_scratch_directory(
    tmp_path, acp_request_log
):
    missing = tmp_path / "not-yet-cloned"
    agent = ACPAgent(acp_command=scripted_acp_command())

    preview_acp_session(
        agent, LocalWorkspace(working_dir=str(missing)), tmp_path / "preview"
    )

    new_session = next(e for e in acp_request_log() if e["method"] == "session/new")
    assert new_session["params"]["cwd"] == str(tmp_path / "preview" / "workspace")
    assert not missing.exists()
    assert list((tmp_path / "preview" / "workspace").iterdir()) == []
