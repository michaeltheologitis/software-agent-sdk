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
- ``--set-error SENTENCE``: answer every ``session/set_config_option`` with an
  internal error (-32603) whose message is ``SENTENCE``.
- ``--auth-required``: answer ``session/new`` with ACP's authentication
  required error (-32000).
- ``--subagents``: play a sub-agent run (ACP schema 1.24.1's unstable sub-agent
  sessions) on each prompt, before the reply. Its children are ``child-a``
  (with a grandchild, ``child-a-1``), ``child-c`` (which cannot be cancelled)
  and ``child-b``; only the root's lines are played when the client did not
  advertise ``clientCapabilities.subagents``.
- ``--cancel-wait SECONDS``: how long ``child-b`` waits for a ``session/cancel``
  naming it (default 0: it does not wait). Cancelled, its cell fails and it
  turns idle with ``stopReason: cancelled``; otherwise both complete.
- ``--transcript PATH``: play a JSONL transcript instead of every behaviour
  above, one JSON-RPC message per line as it appeared on the wire (see
  ``TranscriptPlayer``).
- ``--transcript-interval-ms MS``: wait this long before each
  ``session/update`` the transcript sends (default 0).
- ``--wait-timeout SECONDS``: how long the transcript waits for each client
  message it expects (default 30); past it the script exits non-zero.

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
import sys
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

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
    Cost,
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
INTERNAL_ERROR = -32603
OPTION_ID = "profile"
PROFILES = ("fast", "thorough")
COMMANDS_AFTER_NEW_SESSION_DELAY = 0.05
CONTEXT_WINDOW = 1000
LOG_ENV_VAR = "SCRIPTED_ACP_LOG"
SESSION_UPDATE = CLIENT_METHODS["session_update"]
UNSTABLE_SESSION_UPDATES = frozenset(
    {"subagent_update", "session_message", "session_message_chunk"}
)
DEFAULT_WAIT_TIMEOUT_S: Final[float] = 30.0

ROOT_CELL = "cell-1"
CHILD_A = "child-a"
GRANDCHILD = "child-a-1"
CHILD_B = "child-b"
CHILD_C = "child-c"
CHILD_COST = 0.0004
ROOT_COST = 0.0011

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
        self._child_b_cancelled = asyncio.Event()

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
        if self._args.auth_required:
            raise RequestError.auth_required()
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
        if self._args.set_error is not None:
            raise RequestError(INTERNAL_ERROR, self._args.set_error)
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
        if self._args.subagents:
            assert self.conn is not None
            self._child_b_cancelled.clear()
            await play_subagent_run(
                self.conn,
                session_id,
                advertised=advertises_subagents(self.initialize_params),
                cancel_wait_s=self._args.cancel_wait,
                cancelled=self._child_b_cancelled,
            )
        first_text = next((b.text for b in prompt if b.type == "text"), "")
        await self._update(session_id, update_agent_message_text(first_text))
        cost = Cost(amount=ROOT_COST, currency="USD") if self._args.subagents else None
        await self._update(
            session_id,
            UsageUpdate(
                session_update="usage_update",
                used=len(first_text),
                size=CONTEXT_WINDOW,
                cost=cost,
            ),
        )
        return PromptResponse(stop_reason="end_turn")

    async def close_session(
        self, session_id: str, **kwargs: Any
    ) -> CloseSessionResponse:
        return CloseSessionResponse()

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        if session_id == CHILD_B:
            self._child_b_cancelled.set()

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


def advertises_subagents(initialize_params: dict[str, Any]) -> bool:
    """Whether the client's raw ``initialize`` params advertise ``subagents``."""
    return "subagents" in (initialize_params.get("clientCapabilities") or {})


def _text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _tool_call(call_id: str, title: str) -> dict[str, Any]:
    return {
        "sessionUpdate": "tool_call",
        "toolCallId": call_id,
        "title": title,
        "kind": "execute",
        "status": "in_progress",
    }


def _tool_call_done(call_id: str, status: str, output: str) -> dict[str, Any]:
    return {
        "sessionUpdate": "tool_call_update",
        "toolCallId": call_id,
        "status": status,
        "rawOutput": output,
    }


def _announce(child: str, title: str, cell: str, *, cancel: bool) -> dict[str, Any]:
    update: dict[str, Any] = {
        "sessionUpdate": "subagent_update",
        "sessionId": child,
        "title": title,
        "state": {"state": "running"},
        "_meta": {"openhands": {"parentToolCallId": cell}},
    }
    if cancel:
        update["capabilities"] = {"cancel": {}}
    return update


def _idle(child: str, stop_reason: str) -> dict[str, Any]:
    return {
        "sessionUpdate": "subagent_update",
        "sessionId": child,
        "state": {"state": "idle", "stopReason": stop_reason},
    }


def _message(
    kind: str, message_id: str, sender: str, recipient: str, content: Any
) -> dict[str, Any]:
    return {
        "sessionUpdate": kind,
        "messageId": message_id,
        "senderSessionId": sender,
        "recipientSessionId": recipient,
        "content": content,
    }


def _thought(text: str) -> dict[str, Any]:
    return {"sessionUpdate": "agent_thought_chunk", "content": _text(text)}


def _usage(cost: float) -> dict[str, Any]:
    return {
        "sessionUpdate": "usage_update",
        "used": 10,
        "size": CONTEXT_WINDOW,
        "cost": {"amount": cost, "currency": "USD"},
    }


async def play_subagent_run(
    conn: Connection,
    root_session_id: str,
    *,
    advertised: bool,
    cancel_wait_s: float,
    cancelled: asyncio.Event,
) -> None:
    """Send the generic sub-agent run as raw session/update notifications; only
    the root's lines when the client did not advertise ``subagents``."""

    async def send(session_id: str, update: dict[str, Any]) -> None:
        await conn.send_notification(
            SESSION_UPDATE, {"sessionId": session_id, "update": update}
        )

    root = root_session_id
    await send(root, _tool_call(ROOT_CELL, "Run spawn"))
    if advertised:
        await send(root, _announce(CHILD_A, "Summarize part A", ROOT_CELL, cancel=True))
        await send(
            root,
            _message(
                "session_message",
                "child-a-task",
                root,
                CHILD_A,
                [_text("Summarize part A.")],
            ),
        )
        await send(CHILD_A, _thought("Reading "))
        await send(CHILD_A, _thought("part A."))
        await send(CHILD_A, _tool_call("cell-a1", "Run delegate"))
        await send(
            CHILD_A, _announce(GRANDCHILD, "Check part A", "cell-a1", cancel=True)
        )
        for part in ("Part A ", "checks out."):
            await send(
                GRANDCHILD,
                _message(
                    "session_message_chunk",
                    "child-a-1-answer",
                    GRANDCHILD,
                    CHILD_A,
                    _text(part),
                ),
            )
        await send(CHILD_A, _idle(GRANDCHILD, "end_turn"))
        await send(CHILD_A, _tool_call_done("cell-a1", "completed", "checked"))
        await send(CHILD_A, _usage(CHILD_COST))
        await send(
            CHILD_A,
            _message(
                "session_message",
                "child-a-answer",
                CHILD_A,
                root,
                [_text("Part A: fine.")],
            ),
        )
        await send(root, _idle(CHILD_A, "end_turn"))

        await send(
            root, _announce(CHILD_C, "Summarize part C", ROOT_CELL, cancel=False)
        )
        await send(CHILD_C, _tool_call("cell-c1", "Run count"))
        await send(CHILD_C, _tool_call_done("cell-c1", "completed", "3"))
        await send(root, _idle(CHILD_C, "end_turn"))

        await send(root, _announce(CHILD_B, "Summarize part B", ROOT_CELL, cancel=True))
        await send(CHILD_B, _tool_call("cell-b1", "Run slow"))
        if cancel_wait_s > 0:
            try:
                await asyncio.wait_for(cancelled.wait(), timeout=cancel_wait_s)
            except TimeoutError:
                pass
        if cancelled.is_set():
            await send(CHILD_B, _tool_call_done("cell-b1", "failed", "cancelled"))
            await send(root, _idle(CHILD_B, "cancelled"))
        else:
            await send(CHILD_B, _tool_call_done("cell-b1", "completed", "done"))
            await send(root, _idle(CHILD_B, "end_turn"))
    await send(root, _tool_call_done(ROOT_CELL, "completed", "spawned"))


@dataclass(frozen=True)
class _Wait:
    """Wait for the client's next ``method`` (for session/cancel, naming
    ``session_id``); ``recorded_id`` is the recorded request's id."""

    method: str
    session_id: str | None
    recorded_id: Any


@dataclass(frozen=True)
class _Respond:
    recorded_id: Any
    message: dict[str, Any]


@dataclass(frozen=True)
class _Notify:
    params: dict[str, Any]


@dataclass(frozen=True)
class _Incoming:
    method: str
    params: Any
    response: asyncio.Future[Any] | None


def _inferred_method(response: dict[str, Any], line_number: int) -> str:
    """The request an outgoing-only transcript's response answers, by its shape."""
    result = response.get("result")
    if isinstance(result, dict):
        if "protocolVersion" in result:
            return AGENT_METHODS["initialize"]
        if "stopReason" in result:
            return AGENT_METHODS["session_prompt"]
        if "sessionId" in result:
            return AGENT_METHODS["session_new"]
    raise ValueError(
        f"transcript line {line_number}: cannot tell which request this response "
        "answers; record the client's lines too"
    )


def plan_transcript(
    lines: Sequence[dict[str, Any]],
) -> list[_Wait | _Respond | _Notify]:
    """Turn transcript lines into the player's steps (``TranscriptPlayer``)."""
    client_methods = set(AGENT_METHODS.values())
    outgoing_only = not any(line.get("method") in client_methods for line in lines)
    steps: list[_Wait | _Respond | _Notify] = []
    since_response: list[_Wait | _Respond | _Notify] = []
    for number, line in enumerate(lines, start=1):
        method = line.get("method")
        if method in client_methods:
            params = line.get("params") or {}
            cancelled = params.get("sessionId") if method == "session/cancel" else None
            steps.append(_Wait(method, cancelled, line.get("id")))
        elif method == SESSION_UPDATE:
            (since_response if outgoing_only else steps).append(_Notify(line["params"]))
        elif method is None and "id" in line:
            if outgoing_only:
                wait = _Wait(_inferred_method(line, number), None, line["id"])
                if wait.method == AGENT_METHODS["session_prompt"]:
                    steps += [wait, *since_response]
                else:
                    steps += [*since_response, wait]
                since_response = []
            steps.append(_Respond(line["id"], line))
        else:
            raise ValueError(f"transcript line {number}: unsupported message {line}")
    return steps + since_response


def _root_session_id(lines: Sequence[dict[str, Any]]) -> str | None:
    for line in lines:
        result = line.get("result")
        if isinstance(result, dict) and "sessionId" in result:
            if "stopReason" not in result:
                return result["sessionId"]
    return None


def _log(message: str) -> None:
    print(f"{AGENT_NAME}: {message}", file=sys.stderr, flush=True)


class TranscriptPlayer:
    """Plays one JSONL transcript over one connection.

    - A request or notification whose method the client sends (``initialize``,
      ``session/new``, ``session/prompt``, ``session/cancel``, ...) is a wait
      point: the player waits for the client's next message with that method
      (for ``session/cancel``, naming the same session).
    - A response is sent as the answer to the request its recorded id names.
    - A ``session/update`` notification is sent.

    A transcript of agent lines only (no client lines) gets one inferred wait
    point before each response, by the response's shape: ``initialize``,
    ``session/new``, or ``session/prompt``; a prompt's wait point also comes
    before the notifications since the previous response. When the client did
    not advertise ``subagents``, the three unstable updates and every update for
    a session other than the root are skipped. A client message the transcript
    is not waiting for is logged, and a request gets -32601.
    """

    def __init__(
        self,
        transcript: Sequence[dict[str, Any]],
        *,
        wait_timeout_s: float = DEFAULT_WAIT_TIMEOUT_S,
        interval_s: float = 0.0,
    ) -> None:
        self._steps = plan_transcript(transcript)
        self._root_session_id = _root_session_id(transcript)
        self._wait_timeout_s = wait_timeout_s
        self._interval_s = interval_s
        self._incoming: asyncio.Queue[_Incoming] = asyncio.Queue()
        self._responses: dict[Any, asyncio.Future[Any]] = {}
        self._finished = False
        self.advertised = False

    def on_initialize(self, params: dict[str, Any]) -> None:
        self.advertised = advertises_subagents(params)

    async def handle(
        self,
        method: str,
        params: Any,
        is_notification: bool,
    ) -> Any:
        """The connection's handler: hand a client message to the player."""
        if self._finished:
            return self._refuse(_Incoming(method, params, None), is_notification)
        response = (
            None if is_notification else asyncio.get_running_loop().create_future()
        )
        self._incoming.put_nowait(_Incoming(method, params, response))
        return None if response is None else await response

    async def play(self, conn: Connection) -> None:
        """Walk the transcript once; raise TimeoutError at a missed wait point."""
        for step in self._steps:
            if isinstance(step, _Wait):
                await self._wait(step)
            elif isinstance(step, _Respond):
                self._respond(step)
            elif self._plays(step):
                if self._interval_s:
                    await asyncio.sleep(self._interval_s)
                await conn.send_notification(SESSION_UPDATE, step.params)
        self._finished = True
        while not self._incoming.empty():
            self._reject(self._incoming.get_nowait())

    async def _wait(self, step: _Wait) -> None:
        try:
            async with asyncio.timeout(self._wait_timeout_s):
                while True:
                    incoming = await self._incoming.get()
                    if self._matches(step, incoming):
                        if incoming.response is not None:
                            self._responses[step.recorded_id] = incoming.response
                        return
                    self._reject(incoming)
        except TimeoutError:
            _log(f"no {step.method} within {self._wait_timeout_s:g}s")
            raise

    @staticmethod
    def _matches(step: _Wait, incoming: _Incoming) -> bool:
        if incoming.method != step.method:
            return False
        if step.session_id is None:
            return True
        return (incoming.params or {}).get("sessionId") == step.session_id

    def _respond(self, step: _Respond) -> None:
        response = self._responses.pop(step.recorded_id)
        error = step.message.get("error")
        if error is not None:
            response.set_exception(
                RequestError(error["code"], error.get("message", ""), error.get("data"))
            )
        else:
            response.set_result(step.message.get("result"))

    def _plays(self, step: _Notify) -> bool:
        if self.advertised:
            return True
        update = step.params.get("update") or {}
        return (
            update.get("sessionUpdate") not in UNSTABLE_SESSION_UPDATES
            and step.params.get("sessionId") == self._root_session_id
        )

    def _reject(self, incoming: _Incoming) -> None:
        _log(f"not waiting for {incoming.method}")
        if incoming.response is not None:
            incoming.response.set_exception(
                RequestError.method_not_found(incoming.method)
            )

    def _refuse(self, incoming: _Incoming, is_notification: bool) -> None:
        _log(f"transcript finished; not answering {incoming.method}")
        if not is_notification:
            raise RequestError.method_not_found(incoming.method)


def load_transcript(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
    parser.add_argument("--set-error", default=None)
    parser.add_argument("--auth-required", action="store_true")
    add_subagent_arguments(parser)
    return parser


def add_subagent_arguments(parser: argparse.ArgumentParser) -> None:
    """Add --subagents, --cancel-wait, --transcript, --transcript-interval-ms and
    --wait-timeout."""
    parser.add_argument("--subagents", action="store_true")
    parser.add_argument("--cancel-wait", type=float, default=0.0)
    parser.add_argument("--transcript", type=Path, default=None)
    parser.add_argument("--transcript-interval-ms", type=float, default=0.0)
    parser.add_argument("--wait-timeout", type=float, default=DEFAULT_WAIT_TIMEOUT_S)


async def play_transcript(args: argparse.Namespace) -> None:
    player = TranscriptPlayer(
        load_transcript(args.transcript),
        wait_timeout_s=args.wait_timeout,
        interval_s=args.transcript_interval_ms / 1000,
    )
    conn = await serve(player.handle, on_initialize=player.on_initialize)
    receiving = asyncio.ensure_future(conn.main_loop())
    playing = asyncio.ensure_future(player.play(conn))
    try:
        done, _ = await asyncio.wait(
            {receiving, playing}, return_when=asyncio.FIRST_COMPLETED
        )
        if playing in done:
            playing.result()
            await receiving
    finally:
        for task in (receiving, playing):
            task.cancel()
        await asyncio.shield(conn.close())


async def run(args: argparse.Namespace) -> None:
    if args.transcript is not None:
        await play_transcript(args)
        return
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
