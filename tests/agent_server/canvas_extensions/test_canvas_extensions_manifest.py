"""Tests for the CanvasExtensionManifest model and its field validation."""

from typing import Any

import pytest
from pydantic import ValidationError

from openhands.agent_server.canvas_extensions.manifest import (
    CanvasExtensionContributes,
    CanvasExtensionManifest,
    CanvasExtensionPage,
)


def _manifest(**overrides: Any) -> CanvasExtensionManifest:
    defaults: dict[str, Any] = dict(
        schema_version=1,
        name="my-extension",
        display_name="My Extension",
        version="1.0.0",
        entrypoint="dist/index.js",
    )
    defaults.update(overrides)
    return CanvasExtensionManifest(**defaults)


def test_minimal_manifest():
    manifest = _manifest()
    assert manifest.name == "my-extension"
    assert manifest.description == ""
    assert manifest.contributes.pages == []


def test_manifest_with_page_contribution():
    manifest = _manifest(
        contributes=CanvasExtensionContributes(
            pages=[
                CanvasExtensionPage(
                    id="dashboard", title="Dashboard", path="/dashboard"
                )
            ]
        )
    )
    assert manifest.contributes.pages[0].id == "dashboard"
    assert manifest.contributes.pages[0].title == "Dashboard"
    assert manifest.contributes.pages[0].path == "/dashboard"


def test_manifest_with_multiple_distinct_pages():
    manifest = _manifest(
        contributes=CanvasExtensionContributes(
            pages=[
                CanvasExtensionPage(
                    id="dashboard", title="Dashboard", path="/dashboard"
                ),
                CanvasExtensionPage(
                    id="settings", title="Settings", path="/dashboard/settings"
                ),
            ]
        )
    )
    ids = [p.id for p in manifest.contributes.pages]
    paths = [p.path for p in manifest.contributes.pages]
    assert ids == ["dashboard", "settings"]
    assert paths == ["/dashboard", "/dashboard/settings"]


@pytest.mark.parametrize(
    "name",
    ["CamelCase", "", "has_underscore", "../evil", "-leading-hyphen"],
)
def test_invalid_name_rejected(name: str):
    with pytest.raises(ValidationError, match="Invalid extension name"):
        _manifest(name=name)


@pytest.mark.parametrize(
    "page_id",
    ["CamelCase", "", "has_underscore", "../evil", "with space"],
)
def test_invalid_contribution_id_rejected(page_id: str):
    with pytest.raises(ValidationError, match="Invalid contribution id"):
        CanvasExtensionPage(id=page_id, title="Title", path="/valid")


@pytest.mark.parametrize(
    "path",
    [
        "dashboard",  # missing leading slash
        "",  # empty
        "/",  # root alone — no segment after the slash
        "/../etc/passwd",  # traversal
        "//double-slash",  # empty segment
        "/trailing/",  # trailing slash / empty final segment
        "/Bad-Case",  # uppercase not allowed
        "/foo--bar",  # double hyphen: no segment between them
        "/foo_bar",  # underscore not allowed
    ],
)
def test_invalid_page_path_rejected(path: str):
    with pytest.raises(ValidationError, match="Invalid page path"):
        CanvasExtensionPage(id="valid-id", title="Title", path=path)


def test_valid_multi_segment_page_path_accepted():
    page = CanvasExtensionPage(
        id="settings", title="Settings", path="/dashboard/settings"
    )
    assert page.path == "/dashboard/settings"


def test_duplicate_page_contribution_ids_rejected():
    with pytest.raises(ValidationError, match="Duplicate page contribution id"):
        CanvasExtensionContributes(
            pages=[
                CanvasExtensionPage(id="dup", title="A", path="/a"),
                CanvasExtensionPage(id="dup", title="B", path="/b"),
            ]
        )


def test_duplicate_page_paths_rejected():
    with pytest.raises(ValidationError, match="Duplicate page path"):
        CanvasExtensionContributes(
            pages=[
                CanvasExtensionPage(id="a", title="A", path="/dup"),
                CanvasExtensionPage(id="b", title="B", path="/dup"),
            ]
        )


def test_duplicate_page_contribution_ids_rejected_through_full_manifest():
    """The nested ``contributes.pages`` validator must still fire when built
    from a raw dict (as ``canvas-extension.json`` loads), not just when
    ``CanvasExtensionContributes`` is constructed directly in Python.
    """
    with pytest.raises(ValidationError, match="Duplicate page contribution id"):
        _manifest(
            contributes={
                "pages": [
                    {"id": "dup", "title": "A", "path": "/a"},
                    {"id": "dup", "title": "B", "path": "/b"},
                ]
            }
        )


def test_manifest_round_trips_through_json_dict():
    """``model_validate`` over a full, schema-shaped dict — the actual
    ``canvas-extension.json`` ingestion path — round-trips unchanged.
    """
    payload = {
        "schema_version": 1,
        "name": "my-extension",
        "display_name": "My Extension",
        "version": "1.0.0",
        "description": "Does things",
        "entrypoint": "dist/index.js",
        "contributes": {
            "pages": [
                {"id": "dashboard", "title": "Dashboard", "path": "/dashboard"},
            ]
        },
    }
    manifest = CanvasExtensionManifest.model_validate(payload)
    assert manifest.model_dump() == payload


@pytest.mark.parametrize(
    "entrypoint",
    ["", "/absolute/index.js", "../escape/index.js", "nested/../../escape.js"],
)
def test_invalid_entrypoint_syntax_rejected(entrypoint: str):
    with pytest.raises(ValidationError, match="entrypoint"):
        _manifest(entrypoint=entrypoint)


@pytest.mark.parametrize(
    "missing_field",
    ["schema_version", "name", "display_name", "version", "entrypoint"],
)
def test_missing_required_field_rejected(missing_field: str):
    """Confirms these fields are genuinely required (no silent default),
    catching e.g. an accidental ``= None`` or ``default=""`` creeping in.
    """
    payload: dict[str, Any] = dict(
        schema_version=1,
        name="my-extension",
        display_name="My Extension",
        version="1.0.0",
        entrypoint="dist/index.js",
    )
    del payload[missing_field]
    with pytest.raises(ValidationError, match=missing_field):
        CanvasExtensionManifest(**payload)


def test_schema_version_rejects_non_integer():
    with pytest.raises(ValidationError, match="schema_version"):
        _manifest(schema_version="not-a-number")


# -- Conversation panels -------------------------------------------------------


def _panel(**overrides: Any) -> dict[str, Any]:
    panel: dict[str, Any] = {
        "id": "decompositions",
        "title": "Decompositions",
        "tabs": [{"id": "browse", "title": "Browse", "path": "/"}],
    }
    panel.update(overrides)
    return panel


def _manifest_with_panels(*panels: dict[str, Any], **contributes: Any):
    return CanvasExtensionManifest.model_validate(
        {
            "schema_version": 1,
            "name": "my-extension",
            "display_name": "My Extension",
            "version": "1.0.0",
            "entrypoint": "dist/index.js",
            "contributes": {"conversation_panels": list(panels), **contributes},
        }
    )


def test_a_header_panel_with_tabs_validates():
    manifest = _manifest_with_panels(
        _panel(
            icon="dist/panel.svg",
            tabs=[
                {"id": "browse", "title": "Decompositions", "path": "/"},
                {"id": "create", "title": "Create decomposition", "path": "/create"},
                {"id": "namespaces", "title": "Namespaces", "path": "/namespaces"},
                {"id": "tools", "title": "Tools", "path": "/tools"},
            ],
        )
    )

    (panel,) = manifest.contributes.conversation_panels
    assert (panel.id, panel.title, panel.icon) == (
        "decompositions",
        "Decompositions",
        "dist/panel.svg",
    )
    assert [(t.id, t.path) for t in panel.tabs] == [
        ("browse", "/"),
        ("create", "/create"),
        ("namespaces", "/namespaces"),
        ("tools", "/tools"),
    ]


@pytest.mark.parametrize(
    "panel",
    [
        _panel(id="Not-Kebab"),
        _panel(tabs=[{"id": "Bad_Id", "title": "T", "path": "/"}]),
        _panel(tabs=[{"id": "tab", "title": "T", "path": "relative"}]),
        _panel(tabs=[{"id": "tab", "title": "T", "path": "/Upper"}]),
        _panel(tabs=[{"id": "tab", "title": "T", "path": "/trailing/"}]),
        _panel(
            tabs=[
                {"id": "one", "title": "1", "path": "/same"},
                {"id": "two", "title": "2", "path": "/same"},
            ]
        ),
        _panel(icon="/abs/panel.svg"),
        _panel(icon="../panel.svg"),
        _panel(icon="dist/panel.gif"),
    ],
    ids=[
        "panel-id",
        "tab-id",
        "relative-tab-path",
        "uppercase-tab-path",
        "trailing-slash",
        "duplicate-tab-path",
        "absolute-icon",
        "traversing-icon",
        "icon-type",
    ],
)
def test_a_malformed_panel_makes_the_manifest_invalid(panel: dict[str, Any]):
    with pytest.raises(ValidationError):
        _manifest_with_panels(panel)


@pytest.mark.parametrize(
    "panels, pages",
    [
        ([_panel(id="shared")], [{"id": "shared", "title": "P", "path": "/p"}]),
        (
            [_panel(tabs=[{"id": "shared", "title": "T"}])],
            [{"id": "shared", "title": "P", "path": "/p"}],
        ),
        ([_panel(), _panel(tabs=[{"id": "other", "title": "T"}])], []),
        ([_panel(tabs=[{"id": "decompositions", "title": "T"}])], []),
        ([_panel(), _panel(id="second")], []),
    ],
    ids=[
        "page-and-panel",
        "page-and-tab",
        "two-panels",
        "panel-and-tab",
        "tabs-in-two-panels",
    ],
)
def test_pages_panels_and_tabs_share_one_id_namespace(panels, pages):
    with pytest.raises(ValidationError, match="Duplicate contribution id"):
        _manifest_with_panels(*panels, pages=pages)


def test_a_manifest_without_panels_dumps_exactly_as_before():
    """A local App's backend approval revision hashes this dump."""
    manifest = _manifest(
        contributes=CanvasExtensionContributes(
            pages=[CanvasExtensionPage(id="home", title="Home", path="/home")]
        )
    )

    assert "conversation_panels" not in manifest.model_dump()
    assert manifest.model_dump_json() == (
        '{"schema_version":1,"name":"my-extension","display_name":"My Extension",'
        '"version":"1.0.0","description":"","entrypoint":"dist/index.js",'
        '"contributes":{"pages":[{"id":"home","title":"Home","path":"/home"}]}}'
    )
