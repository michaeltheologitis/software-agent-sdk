"""Live checks of ACP sub-agent sessions against a real ACP agent.

Deselected by default with the ``acp_live`` marker, and skipped unless the
environment names the agent and a prompt that makes it spawn sub-agents:

- ``OPENHANDS_ACP_LIVE_AGENT_COMMAND``: the agent's command, shell-split.
- ``OPENHANDS_ACP_LIVE_SUBAGENTS_PROMPT``: a prompt that makes that agent spawn
  sub-agents, at least one with a child of its own.

The agent runs with ``acp_subagents=True``.
"""

from __future__ import annotations

import os
import shlex
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from openhands.sdk.agent.acp_agent import ACPAgent
from openhands.sdk.conversation import LocalConversation
from openhands.sdk.event import (
    ACPSessionTextEvent,
    ACPSubagentEvent,
    ACPToolCallEvent,
    ActionEvent,
)
from openhands.sdk.tool.builtins.finish import FinishAction
from tests.conftest import subagent_snapshots, wait_until


pytestmark = pytest.mark.acp_live

LIVE_AGENT_COMMAND = os.environ.get("OPENHANDS_ACP_LIVE_AGENT_COMMAND", "")
SUBAGENTS_PROMPT = os.environ.get("OPENHANDS_ACP_LIVE_SUBAGENTS_PROMPT", "")
# Long enough that a child's segment appearing in the answer is not a
# coincidence of short phrases.
QUOTED_TEXT_MIN_CHARS = 40

requires_live_agent = pytest.mark.skipif(
    not (LIVE_AGENT_COMMAND and SUBAGENTS_PROMPT),
    reason=(
        "OPENHANDS_ACP_LIVE_AGENT_COMMAND and OPENHANDS_ACP_LIVE_SUBAGENTS_PROMPT "
        "are not both set"
    ),
)


@pytest.fixture
def live_conversation(tmp_path: Path) -> Iterator[Callable[..., LocalConversation]]:
    conversations: list[LocalConversation] = []
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def _conversation(**kwargs: Any) -> LocalConversation:
        agent = ACPAgent(
            acp_command=shlex.split(LIVE_AGENT_COMMAND), acp_subagents=True
        )
        conv = LocalConversation(
            agent,
            workspace=str(workspace),
            persistence_dir=str(tmp_path / "conversations"),
            visualizer=None,
            **kwargs,
        )
        conversations.append(conv)
        return conv

    yield _conversation
    for conv in conversations:
        conv.close()


@requires_live_agent
def test_live_agent_tree_is_well_formed(live_conversation):
    conv = live_conversation()
    conv.send_message(SUBAGENTS_PROMPT)
    conv.run()

    def settled() -> dict[str, ACPSubagentEvent] | None:
        snapshots = subagent_snapshots(conv.state.events)
        done = snapshots and all(s.state != "running" for s in snapshots.values())
        return snapshots if done else None

    snapshots = wait_until(settled, timeout=30)
    events = list(conv.state.events)
    calls = {
        (e.acp_session_id, e.tool_call_id)
        for e in events
        if isinstance(e, ACPToolCallEvent)
    }
    assert any(s.parent_session_id is not None for s in snapshots.values()), (
        "the prompt should make a sub-agent spawn a child of its own"
    )
    for child, snapshot in snapshots.items():
        assert snapshot.parent_session_id in (None, *snapshots), child
        if snapshot.parent_tool_call_id is not None:
            assert (snapshot.parent_session_id, snapshot.parent_tool_call_id) in calls

    [answer] = [
        e.action.message
        for e in events
        if isinstance(e, ActionEvent) and isinstance(e.action, FinishAction)
    ]
    child_texts = [
        e.text
        for e in events
        if isinstance(e, ACPSessionTextEvent) and len(e.text) >= QUOTED_TEXT_MIN_CHARS
    ]
    assert not [text for text in child_texts if text in answer]

    agent = conv.agent
    assert isinstance(agent, ACPAgent) and agent._client is not None
    root_cost = agent._client._last_cost_by_session.get(agent._session_id or "", 0.0)
    combined = conv.state.stats.get_combined_metrics().accumulated_cost
    assert combined == pytest.approx(root_cost)
