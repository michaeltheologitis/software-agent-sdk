"""Shared fixtures for canvas extension tests."""

import json
import socket
from pathlib import Path
from typing import Any

import pytest

from openhands.agent_server.canvas_extensions.manifest import MANIFEST_FILENAME


def write_extension(
    directory: Path,
    name: str = "my-extension",
    version: str = "1.0.0",
    display_name: str = "My Extension",
    description: str = "",
    entrypoint: str = "dist/index.js",
    pages: list[dict[str, str]] | None = None,
    conversation_panels: list[dict[str, Any]] | None = None,
) -> Path:
    """Write a valid, loadable canvas extension package to *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "name": name,
        "display_name": display_name,
        "version": version,
        "description": description,
        "entrypoint": entrypoint,
    }
    if pages is not None:
        manifest["contributes"] = {"pages": pages}
    if conversation_panels is not None:
        manifest.setdefault("contributes", {})["conversation_panels"] = (
            conversation_panels
        )
    (directory / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    entry_file = directory / entrypoint
    entry_file.parent.mkdir(parents=True, exist_ok=True)
    entry_file.write_text("console.log('ok')")
    return directory


@pytest.fixture
def dead_http_proxy(monkeypatch: pytest.MonkeyPatch) -> str:
    """Point every proxy variable at a closed port, exempting nothing.

    What a macOS system proxy does to loopback traffic, whose default
    exceptions do not include 127.0.0.1.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        dead = f"http://127.0.0.1:{sock.getsockname()[1]}"
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(name, dead)
        monkeypatch.setenv(name.lower(), dead)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    return dead


@pytest.fixture
def extension_dir(tmp_path: Path) -> Path:
    return write_extension(tmp_path / "source" / "my-extension")


@pytest.fixture
def installed_dir(tmp_path: Path) -> Path:
    return tmp_path / "installed"
