"""A scripted ACP agent for tests: deterministic, generic, and model-free.

Run it as ``[sys.executable, "tests/fixtures/acp/scripted_agent.py", *flags]``.

It offers one ``select`` config option, ``profile`` (``fast`` or ``thorough``),
and slash commands that depend on the profile: ``fast`` offers ``summarize``;
``thorough`` offers ``summarize`` and ``compare``. The commands are sent right
after ``session/new`` answers, and again before a ``session/set_config_option``
answers. The first prompt clears the commands and fixes the profile.

Flags:

- ``--no-close``: do not advertise ``sessionCapabilities.close``.
- ``--no-commands``: never send ``available_commands_update``.
- ``--slow-set SECONDS``: wait before answering ``session/set_config_option``.
- ``--sessions-file PATH``: keep sessions in a JSON file, so that a later
  process can ``session/load`` them.

When the ``SCRIPTED_ACP_LOG`` environment variable names a file, every request
and notification the agent receives is appended to it as one JSON line,
``{"method": ..., "params": ...}``, in arrival order.

The agent is served through ``acp.connection.Connection`` with
``build_agent_router`` behind a small tap, rather than ``acp.run_agent``, so the
script can send raw notifications and read ``initialize``'s raw params, which
``AgentSideConnection`` cannot.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from acp import Agent, RequestError
from acp.agent.router import build_agent_router
from acp.connection import Connection, JsonValue, MethodHandler
from acp.core import DEFAULT_STDIO_BUFFER_LIMIT_BYTES
from acp.helpers import update_agent_message_text
from acp.meta import AGENT_METHODS, CLIENT_METHODS
from acp.schema import (
    AgentCapabilities,
    AvailableCommand,
    AvailableCommandInput,
    AvailableCommandsUpdate,
    CloseSessionResponse,
    ConfigOptionUpdate,
    Implementation,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    SessionCapabilities,
    SessionCloseCapabilities,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SessionNotification,
    SetSessionConfigOptionResponse,
    UnstructuredCommandInput,
    UsageUpdate,
)
from acp.stdio import stdio_streams
from acp.utils import notify_model


AGENT_NAME = "scripted-acp-agent"
INVALID_PARAMS = -32602
OPTION_ID = "profile"
PROFILES = ("fast", "thorough")
COMMANDS_AFTER_NEW_SESSION_DELAY = 0.05
CONTEXT_WINDOW = 1000
LOG_ENV_VAR = "SCRIPTED_ACP_LOG"

SUMMARIZE = AvailableCommand(name="summarize", description="Summarize the input")
COMPARE = AvailableCommand(
    name="compare",
    description="Compare two things",
    input=AvailableCommandInput(UnstructuredCommandInput(hint="what to compare")),
)
COMMANDS = {"fast": [SUMMARIZE], "thorough": [SUMMARIZE, COMPARE]}


def _invalid_params(sentence: str) -> RequestError:
    return RequestError(INVALID_PARAMS, sentence)


class ScriptedAgent:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self.conn: Connection | None = None
        self.initialize_params: dict[str, Any] = {}
        self._sessions_file: Path | None = args.sessions_file
        self._sessions: dict[str, dict[str, Any]] = self._read_sessions()
        self._background: set[asyncio.Task[None]] = set()

    async def initialize(
        self, protocol_version: int, **kwargs: Any
    ) -> InitializeResponse:
        close = None if self._args.no_close else SessionCloseCapabilities()
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_capabilities=AgentCapabilities(
                load_session=True,
                session_capabilities=SessionCapabilities(close=close),
            ),
            agent_info=Implementation(name=AGENT_NAME, version="1.0.0"),
        )

    async def new_session(self, cwd: str, **kwargs: Any) -> NewSessionResponse:
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = {"profile": PROFILES[0], "prompted": False}
        self._write_sessions()
        self._send_commands_soon(session_id)
        return NewSessionResponse(
            session_id=session_id, config_options=[self._option(session_id)]
        )

    async def load_session(
        self, cwd: str, session_id: str, **kwargs: Any
    ) -> LoadSessionResponse:
        if session_id not in self._sessions:
            raise _invalid_params(f"unknown session '{session_id}'")
        if not self._sessions[session_id]["prompted"]:
            self._send_commands_soon(session_id)
        return LoadSessionResponse(config_options=[self._option(session_id)])

    async def set_config_option(
        self, config_id: str, session_id: str, value: str | bool, **kwargs: Any
    ) -> SetSessionConfigOptionResponse:
        if self._args.slow_set:
            await asyncio.sleep(self._args.slow_set)
        session = self._session(session_id)
        if config_id != OPTION_ID:
            raise _invalid_params(f"unknown option '{config_id}'")
        if value not in PROFILES:
            raise _invalid_params(f"unknown profile '{value}'")
        if session["prompted"] and value != session["profile"]:
            raise _invalid_params(
                "profile is fixed once the session has started "
                f"(it is '{session['profile']}')"
            )
        session["profile"] = value
        self._write_sessions()
        await self._send_commands(session_id)
        return SetSessionConfigOptionResponse(config_options=[self._option(session_id)])

    async def prompt(
        self, session_id: str, prompt: list[Any], **kwargs: Any
    ) -> PromptResponse:
        session = self._session(session_id)
        if not session["prompted"]:
            session["prompted"] = True
            self._write_sessions()
            await self._send_commands(session_id)
            await self._update(
                session_id,
                ConfigOptionUpdate(
                    session_update="config_option_update",
                    config_options=[self._option(session_id)],
                ),
            )
        first_text = next((b.text for b in prompt if b.type == "text"), "")
        await self._update(session_id, update_agent_message_text(first_text))
        await self._update(
            session_id,
            UsageUpdate(
                session_update="usage_update", used=len(first_text), size=CONTEXT_WINDOW
            ),
        )
        return PromptResponse(stop_reason="end_turn")

    async def close_session(
        self, session_id: str, **kwargs: Any
    ) -> CloseSessionResponse:
        return CloseSessionResponse()

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        return None

    def _session(self, session_id: str) -> dict[str, Any]:
        if session_id not in self._sessions:
            raise _invalid_params(f"unknown session '{session_id}'")
        return self._sessions[session_id]

    def _option(self, session_id: str) -> SessionConfigOptionSelect:
        session = self._sessions[session_id]
        values = [session["profile"]] if session["prompted"] else list(PROFILES)
        return SessionConfigOptionSelect(
            type="select",
            id=OPTION_ID,
            name="Profile",
            current_value=session["profile"],
            options=[SessionConfigSelectOption(value=v, name=v) for v in values],
        )

    async def _send_commands(self, session_id: str) -> None:
        if self._args.no_commands:
            return
        session = self._sessions[session_id]
        commands = [] if session["prompted"] else COMMANDS[session["profile"]]
        await self._update(
            session_id,
            AvailableCommandsUpdate(
                session_update="available_commands_update",
                available_commands=commands,
            ),
        )

    def _send_commands_soon(self, session_id: str) -> None:
        async def send() -> None:
            await asyncio.sleep(COMMANDS_AFTER_NEW_SESSION_DELAY)
            await self._send_commands(session_id)

        task = asyncio.get_running_loop().create_task(send())
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _update(self, session_id: str, update: Any) -> None:
        assert self.conn is not None
        await notify_model(
            self.conn,
            CLIENT_METHODS["session_update"],
            SessionNotification(session_id=session_id, update=update),
        )

    def _read_sessions(self) -> dict[str, dict[str, Any]]:
        if self._sessions_file is None or not self._sessions_file.exists():
            return {}
        return json.loads(self._sessions_file.read_text())

    def _write_sessions(self) -> None:
        if self._sessions_file is not None:
            self._sessions_file.write_text(json.dumps(self._sessions))


def _log_request(method: str, params: JsonValue | None) -> None:
    log_path = os.environ.get(LOG_ENV_VAR)
    if not log_path:
        return
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(json.dumps({"method": method, "params": params}) + "\n")


async def serve(
    handler: MethodHandler,
    *,
    on_initialize: Callable[[dict[str, Any]], None],
) -> Connection:
    """Serve ``handler`` over stdio through ``acp.connection.Connection``.

    A tap in front of ``handler`` writes the request log and passes
    ``initialize``'s raw params to ``on_initialize``.
    """

    async def tap(
        method: str, params: JsonValue | None, is_notification: bool
    ) -> JsonValue | None:
        _log_request(method, params)
        if method == AGENT_METHODS["initialize"] and isinstance(params, dict):
            on_initialize(params)
        return await handler(method, params, is_notification)

    reader, writer = await stdio_streams(limit=DEFAULT_STDIO_BUFFER_LIMIT_BYTES)
    return Connection(tap, writer, reader, listening=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-close", action="store_true")
    parser.add_argument("--no-commands", action="store_true")
    parser.add_argument("--slow-set", type=float, default=0.0)
    parser.add_argument("--sessions-file", type=Path, default=None)
    return parser


async def run(args: argparse.Namespace) -> None:
    agent = ScriptedAgent(args)
    conn = await serve(
        build_agent_router(cast(Agent, agent), use_unstable_protocol=True),
        on_initialize=agent.initialize_params.update,
    )
    agent.conn = conn
    try:
        await conn.main_loop()
    finally:
        await asyncio.shield(conn.close())


def main() -> None:
    asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
