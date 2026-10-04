"""The ACP routes: previewing an agent's session controls and setting options.

The ACP agent behind them is the scripted test agent, run as a real process.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import APIRouter, FastAPI

import openhands.sdk.agent.acp_agent as acp_agent_module
from openhands.agent_server.acp_router import acp_router, conversation_acp_router
from openhands.agent_server.api import _add_exception_handlers
from openhands.agent_server.config import Config
from openhands.agent_server.conversation_router import conversation_router
from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.event_router import event_router
from openhands.agent_server.event_service import RunSlot
from openhands.agent_server.persistence import reset_stores
from openhands.agent_server.persistence.store import get_agent_profile_store
from openhands.agent_server.server_details_router import build_server_info
from openhands.sdk import LLM, Agent
from openhands.sdk.agent.acp_agent import ACPAgent
from openhands.sdk.agent.acp_models import ACPConfigOption
from openhands.sdk.profiles.agent_profile import ACPAgentProfile
from tests.conftest import SCRIPTED_ACP_AGENT, scripted_acp_command


SUMMARIZE = {"name": "summarize", "description": "Summarize the input", "input": None}
COMPARE = {
    "name": "compare",
    "description": "Compare two things",
    "input": {"hint": "what to compare"},
}


def profile_option(current: str, *values: str) -> dict[str, Any]:
    return {
        "id": "profile",
        "name": "Profile",
        "type": "select",
        "current_value": current,
        "description": None,
        "category": None,
        "options": [
            {"value": v, "name": v, "description": None, "group": None} for v in values
        ],
    }


# The events search matches an event's qualified class name.
CONTROLS_EVENT_KIND = "openhands.sdk.event.acp_session_controls.ACPSessionControlsEvent"
THOROUGH = {
    "available_commands": [SUMMARIZE, COMPARE],
    "config_options": [profile_option("thorough", "fast", "thorough")],
}


class Server:
    def __init__(
        self, client: httpx.AsyncClient, service: ConversationService, root: Path
    ) -> None:
        self.client = client
        self.service = service
        self.root = root
        self.workspace = {"working_dir": str(root / "workspace")}

    def previews_left_behind(self) -> list[Path]:
        return list((self.root / "conversations").glob("preview-*"))

    async def start(self, **payload: Any) -> UUID:
        response = await self.client.post(
            "/api/conversations",
            json={"workspace": self.workspace, "autotitle": False, **payload},
        )
        assert response.status_code == 201, response.text
        return UUID(response.json()["id"])

    async def start_and_run(self, **payload: Any) -> UUID:
        conversation_id = await self.start(
            initial_message={"content": [{"type": "text", "text": "hello"}]}, **payload
        )
        event_service = await self.service.get_event_service(conversation_id)
        assert event_service is not None
        await event_service.wait_for_run_completion(30)
        return conversation_id

    async def set_option(
        self, conversation_id: UUID, config_id: str, value: Any
    ) -> httpx.Response:
        return await self.client.post(
            f"/api/conversations/{conversation_id}/acp/config-options",
            json={"config_id": config_id, "value": value},
        )

    async def newest_controls(self, conversation_id: UUID) -> dict[str, Any] | None:
        response = await self.client.get(
            f"/api/conversations/{conversation_id}/events/search",
            params={
                "kind": CONTROLS_EVENT_KIND,
                "sort_order": "TIMESTAMP_DESC",
                "limit": 1,
            },
        )
        items = response.json()["items"]
        return items[0] if items else None


@asynccontextmanager
async def serving(
    root: Path, *, max_concurrent_runs: int | None = None, runtime: str = "local"
) -> AsyncIterator[Server]:
    (root / "workspace").mkdir(exist_ok=True)
    kwargs: dict[str, Any] = {}
    if max_concurrent_runs is not None:
        kwargs["max_concurrent_runs"] = max_concurrent_runs
    async with ConversationService(
        conversations_dir=root / "conversations", **kwargs
    ) as service:
        app = FastAPI()
        config = Config(static_files_path=None, session_api_keys=[], secret_key=None)
        app.state.config = config.model_copy(update={"conversation_runtime": runtime})
        app.state.conversation_service = service
        api = APIRouter(prefix="/api")
        api.include_router(event_router)
        api.include_router(conversation_router)
        api.include_router(conversation_acp_router)
        api.include_router(acp_router)
        app.include_router(api)
        _add_exception_handlers(app)
        # Answer an unhandled error with the 500 handler's body, as a deployed
        # server does, instead of raising it into the test.
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", timeout=60
        ) as client:
            yield Server(client, service, root)


@pytest.fixture
def stores(tmp_path, monkeypatch):
    reset_stores()
    monkeypatch.setenv("OH_PERSISTENCE_DIR", str(tmp_path / "persistence"))
    yield
    reset_stores()


@pytest.fixture
async def server(tmp_path, stores) -> AsyncIterator[Server]:
    async with serving(tmp_path) as running:
        yield running


def scripted_settings(*flags: str) -> dict[str, Any]:
    return {
        "agent_kind": "acp",
        "acp_server": "custom",
        "acp_command": scripted_acp_command(*flags),
    }


def scripted_agent(*flags: str, **fields: Any) -> dict[str, Any]:
    agent = ACPAgent(acp_command=scripted_acp_command(*flags), **fields)
    return agent.model_dump(mode="json")


def scripted_profile_id(**fields: Any) -> str:
    profile = ACPAgentProfile(
        name="scripted",
        acp_server="custom",
        acp_command=sys.executable,
        acp_args=[str(SCRIPTED_ACP_AGENT)],
        **fields,
    )
    get_agent_profile_store().save(profile)
    return str(profile.id)


def plain_agent() -> dict[str, Any]:
    return Agent(llm=LLM(model="gpt-4o", usage_id="llm"), tools=[]).model_dump(
        mode="json"
    )


# -- The preview ----------------------------------------------------------------


@pytest.mark.parametrize("named_by", ["agent", "agent_settings", "agent_profile_id"])
async def test_the_preview_answers_for_each_way_of_naming_the_agent(server, named_by):
    payload = {
        "agent": {"agent": scripted_agent()},
        "agent_settings": {"agent_settings": scripted_settings()},
        "agent_profile_id": {"agent_profile_id": scripted_profile_id()},
    }[named_by]

    response = await server.client.post(
        "/api/acp/preview",
        json={
            "workspace": server.workspace,
            "acp_config_options": {"profile": "thorough"},
            **payload,
        },
    )

    assert response.status_code == 200, response.text
    assert response.json() == THOROUGH
    assert server.previews_left_behind() == []


@pytest.mark.parametrize(
    "agent, options, status, detail",
    [
        (scripted_agent(), {"profile": "turbo"}, 422, "unknown profile 'turbo'"),
        (scripted_agent(acp_startup_timeout=0.001), {}, 504, None),
        (ACPAgent(acp_command=["/nonexistent/acp-agent"]).model_dump(), {}, 502, None),
        (plain_agent(), {}, 400, "preview needs an ACP agent"),
        (plain_agent(), {"profile": "fast"}, 422, None),
    ],
    ids=[
        "refused-value",
        "startup-timeout",
        "spawn-error",
        "not-acp",
        "values-not-acp",
    ],
)
async def test_the_preview_maps_each_failure_to_its_status(
    server, agent, options, status, detail
):
    response = await server.client.post(
        "/api/acp/preview",
        json={
            "workspace": server.workspace,
            "agent": agent,
            "acp_config_options": options,
        },
    )

    assert response.status_code == status, response.text
    if detail is not None:
        assert response.json()["detail"] == detail
    assert server.previews_left_behind() == []


async def test_the_preview_answers_an_authentication_failure_with_502_not_401(server):
    response = await server.client.post(
        "/api/acp/preview",
        json={
            "workspace": server.workspace,
            "agent": scripted_agent("--auth-required"),
        },
    )

    assert response.status_code == 502
    # The agent-server's handler for a 5xx moves the route's detail into
    # "exception".
    assert response.json() == {
        "detail": "Internal Server Error",
        "exception": "502: [-32000] Authentication required",
    }
    assert server.previews_left_behind() == []


async def test_the_preview_of_an_unknown_profile_is_not_found(server):
    response = await server.client.post(
        "/api/acp/preview",
        json={"workspace": server.workspace, "agent_profile_id": str(uuid4())},
    )

    assert response.status_code == 404


async def test_the_preview_of_a_profile_with_a_dangling_mcp_reference_is_refused(
    server,
):
    profile_id = scripted_profile_id(mcp_server_refs=["no-such-server"])

    response = await server.client.post(
        "/api/acp/preview",
        json={"workspace": server.workspace, "agent_profile_id": profile_id},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["dangling_mcp_server_refs"] == ["no-such-server"]
    assert server.previews_left_behind() == []


async def test_the_preview_holds_a_run_slot(tmp_path, stores):
    async with serving(tmp_path, max_concurrent_runs=1) as limited:
        payload = {"workspace": limited.workspace, "agent": scripted_agent()}
        with await RunSlot.acquire(limited.service._run_semaphore):
            busy = await limited.client.post("/api/acp/preview", json=payload)
        free = await limited.client.post("/api/acp/preview", json=payload)

    assert busy.status_code == 429
    assert free.status_code == 200


async def test_the_preview_is_unavailable_in_the_docker_runtime(tmp_path, stores):
    async with serving(tmp_path, runtime="docker") as docker:
        response = await docker.client.post(
            "/api/acp/preview",
            json={"workspace": docker.workspace, "agent": scripted_agent()},
        )

    assert response.status_code == 501


# -- Starting with option values --------------------------------------------------


async def test_the_start_folds_option_values_into_the_agent_only(server):
    conversation_id = await server.start(
        agent_settings=scripted_settings(), acp_config_options={"profile": "thorough"}
    )

    conversation_dir = server.root / "conversations" / conversation_id.hex
    base_state = json.loads((conversation_dir / "base_state.json").read_text())
    meta = json.loads((conversation_dir / "meta.json").read_text())
    assert base_state["agent"]["acp_config_options"] == {"profile": "thorough"}
    assert "acp_config_options" not in meta


async def test_a_started_session_reports_the_chosen_value_and_cleared_commands(server):
    conversation_id = await server.start_and_run(
        agent=scripted_agent(), acp_config_options={"profile": "thorough"}
    )

    async def settled() -> dict[str, Any] | None:
        newest = await server.newest_controls(conversation_id)
        narrowed = newest and len(newest["config_options"][0]["options"]) == 1
        return newest if narrowed else None

    async with asyncio.timeout(10):
        while (newest := await settled()) is None:
            await asyncio.sleep(0.05)
    # The events API leaves out null fields; compare the parsed options.
    assert newest["available_commands"] == []
    assert [ACPConfigOption.model_validate(o) for o in newest["config_options"]] == [
        ACPConfigOption.model_validate(profile_option("thorough", "thorough"))
    ]


@pytest.mark.parametrize(
    "agent, options",
    [(plain_agent(), {"profile": "fast"}), (scripted_agent(), {"model": "big"})],
    ids=["not-acp", "model-option"],
)
async def test_the_start_refuses_option_values_it_cannot_apply(server, agent, options):
    response = await server.client.post(
        "/api/conversations",
        json={
            "workspace": server.workspace,
            "agent": agent,
            "acp_config_options": options,
        },
    )

    assert response.status_code == 422


# -- Setting an option ---------------------------------------------------------------


async def test_a_set_before_the_start_is_kept_for_it(server):
    conversation_id = await server.start(agent=scripted_agent())

    response = await server.set_option(conversation_id, "profile", "thorough")

    assert response.status_code == 200
    assert response.json() == {
        "applied": False,
        "controls": {"available_commands": [], "config_options": []},
    }
    conversation_dir = server.root / "conversations" / conversation_id.hex
    base_state = json.loads((conversation_dir / "base_state.json").read_text())
    assert base_state["agent"]["acp_config_options"] == {"profile": "thorough"}


async def test_a_live_set_answers_with_the_agents_controls(server):
    conversation_id = await server.start_and_run(agent=scripted_agent())

    response = await server.set_option(conversation_id, "profile", "fast")

    assert response.status_code == 200
    assert response.json() == {
        "applied": True,
        "controls": {
            "available_commands": [],
            "config_options": [profile_option("fast", "fast")],
        },
    }


async def test_a_refusal_passes_the_agents_sentence_through(server):
    conversation_id = await server.start_and_run(agent=scripted_agent())

    response = await server.set_option(conversation_id, "profile", "thorough")

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "profile is fixed once the session has started (it is 'fast')"
    )


@pytest.mark.parametrize(
    "agent, config_id, status",
    [
        (plain_agent(), "profile", 400),
        (scripted_agent(), "model", 400),
        (scripted_agent(), "", 422),
    ],
    ids=["not-acp", "model-option", "empty-id"],
)
async def test_a_set_that_is_not_for_this_route_is_refused(
    server, agent, config_id, status
):
    conversation_id = await server.start(agent=agent)

    response = await server.set_option(conversation_id, config_id, "x")

    assert response.status_code == status


async def test_a_set_on_a_service_that_closed_after_its_lookup_is_a_bad_request(
    server, monkeypatch
):
    conversation_id = await server.start(agent=scripted_agent())
    event_service = await server.service.get_event_service(conversation_id)
    assert event_service is not None
    await event_service.close()

    # A looked-up service can close (idle eviction, a delete) before the set
    # reaches it; a later lookup would start it again.
    async def the_closed_service(_conversation_id: UUID):
        return event_service

    monkeypatch.setattr(server.service, "get_event_service", the_closed_service)

    response = await server.set_option(conversation_id, "profile", "fast")

    assert response.status_code == 400
    assert response.json()["detail"] == "inactive_service"


async def test_a_set_on_an_unknown_conversation_is_not_found(server):
    response = await server.set_option(uuid4(), "profile", "fast")

    assert response.status_code == 404


async def test_an_internal_error_from_the_agent_is_a_500_carrying_its_message_unmasked(
    server,
):
    sentence = "lost the backend key sk-scripted-1234"
    conversation_id = await server.start_and_run(
        agent=scripted_agent("--set-error", sentence),
        secrets={"BACKEND_KEY": {"kind": "StaticSecret", "value": "sk-scripted-1234"}},
    )

    response = await server.set_option(conversation_id, "profile", "fast")

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal Server Error"
    assert response.json()["exception"] == sentence


async def test_a_set_the_agent_does_not_answer_times_out(server, monkeypatch):
    conversation_id = await server.start_and_run(
        agent=scripted_agent("--slow-set", "30")
    )
    monkeypatch.setattr(acp_agent_module, "_ACP_CONFIG_OPTION_TIMEOUT", 0.5)

    response = await server.set_option(conversation_id, "profile", "fast")

    assert response.status_code == 504


# -- Feature detection ----------------------------------------------------------------


def test_server_info_announces_acp_session_controls():
    assert "acp_session_controls_v1" in build_server_info().capabilities
