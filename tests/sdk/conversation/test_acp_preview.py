"""preview_acp_session: what an ACP agent offers before any conversation exists."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import psutil
import pytest

from openhands.sdk.agent.acp_agent import ACPAgent
from openhands.sdk.agent.acp_models import ACPSessionControls
from openhands.sdk.conversation import LocalConversation
from openhands.sdk.conversation.acp_preview import (
    PREVIEW_COMMANDS_WAIT_SECONDS,
    ACPPreviewError,
    preview_acp_session,
)
from openhands.sdk.event import ACPSessionControlsEvent
from openhands.sdk.workspace import LocalWorkspace
from tests.conftest import SCRIPTED_ACP_AGENT, scripted_acp_command


def wait_until(condition: Callable[[], Any], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.02)


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


@pytest.fixture
def started_controls(
    tmp_path: Path, workspace: LocalWorkspace
) -> Iterator[Callable[..., ACPSessionControls]]:
    """The controls a started conversation reports before its first prompt."""
    conversations: list[LocalConversation] = []

    def _started(**fields: Any) -> ACPSessionControls:
        conv = LocalConversation(
            ACPAgent(acp_command=scripted_acp_command(), **fields),
            workspace=workspace,
            persistence_dir=str(tmp_path / "conversations"),
            visualizer=None,
        )
        conversations.append(conv)
        # No message: the session starts, and the run ends without a prompt.
        conv.run()

        def latest() -> ACPSessionControls | None:
            events = [
                e for e in conv.state.events if isinstance(e, ACPSessionControlsEvent)
            ]
            return events[-1].controls if events else None

        wait_until(lambda: (controls := latest()) and controls.available_commands)
        controls = latest()
        assert controls is not None
        return controls

    yield _started
    for conv in conversations:
        conv.close()


@pytest.mark.parametrize(
    "values, commands",
    [
        ({}, ["summarize"]),
        ({"profile": "fast"}, ["summarize"]),
        ({"profile": "thorough"}, ["summarize", "compare"]),
    ],
)
def test_the_preview_equals_the_started_session_before_its_first_prompt(
    preview, started_controls, values, commands
):
    previewed = preview(acp_config_options=values)

    assert [c.name for c in previewed.available_commands] == commands
    assert previewed == started_controls(acp_config_options=values)


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
